import os
import sys
import tqdm
import yaml
import hydra
# import wandb
import torch
import numpy as np
from string import Template
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, '../')
from src.manifolds import make_folder, set_logger
from pytorch.data import ESC50, FSD50K, SimpleSampler
from pytorch.models import Vgg8
from pytorch.utils import loss_fn
from pytorch.eval_kit import Metrics

# Automatically select a spare gpu
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')
# Set configuration file
config_path = '../cfg'
config_name = 'canon.yaml'
# Monitor the model performance, see more in https://github.com/wandb/client
with open(os.path.join(config_path, config_name)) as f:
    yaml_data = yaml.safe_load(f)
t = Template(yaml_data['OUTPUTS']['DIR'].replace('.', '_'))
output_dir = t.substitute(DATASOURCE_NAME=yaml_data['defaults'][0]['DATASOURCE'],
                          TRAINER_LEARNING_RATE=float(yaml_data['TRAINER']['LEARNING_RATE']),
                          TRAINER_BATCH_SIZE=yaml_data['TRAINER']['BATCH_SIZE'])
output_dir = os.path.join(output_dir, f"{time.strftime('%b-%d_%H-%M', time.localtime())}")
# os.environ['WANDB_DIR'] = output_dir
# make_folder(os.environ['WANDB_DIR'])
# wandb.init(project="AudioTagging", entity="jinhua_liang")
# Multiple processing
torch.multiprocessing.set_sharing_strategy('file_system')
# Set root logger
logger = set_logger(log_dir=output_dir)


def train(cfg: OmegaConf) -> None:
    """ Training process."""
    # Dir & Params
    dataset_dir = cfg['DATASOURCE']['DATASET_DIR']
    output_dir = cfg['OUTPUTS']['DIR']
    ckpt_dir = os.path.join(output_dir, 'ckpts')

    device = torch.device('cuda') if cfg['TRAINER']['CUDA'] and torch.cuda.is_available() else torch.device('cpu')
    num_class = cfg['DATASOURCE']['NUM_CLASS']
    K = cfg['DATASOURCE']['NUM_FOLD']  # k-folds cross validation, k = `NUM_FOLD` decided by the curation of datasets
    batch_size = cfg['TRAINER']['BATCH_SIZE']
    epochs = cfg['TRAINER']['EPOCHS']
    learning_rate = cfg['TRAINER']['LEARNING_RATE']
    # Create label split to maintain some classes for few-shot evaluation if applicable
    if cfg['DATASOURCE']['LABEL_SPLIT']:
        # todo: select different split methods
        nvl_cls = [3, 8, 5, 10, 17, 14, 29, 27, 24, 38, 35, 32, 47, 42, 46]
        logger.info(f"Novel split contains {nvl_cls}")
        slt_cls = [x for x in range(num_class) if x not in nvl_cls]
    else:
        slt_cls = [x for x in range(num_class)]
    logger.info(f"Now training on {slt_cls}")
    # Cross-validation
    history = [{} for _ in range(K)]  # history[0...4] corresponds fold1...5
    for fold in range(1, K + 1):
        logger.info('====================================================================================')
        logger.info(f"Experiments: {fold}/{K}")
        if cfg['DATASOURCE']['NAME'] == 'esc50':
            # Cross validation
            train_fold = [x for x in range(1, 6) if x != fold]
            eval_fold = [fold]
            # Set dataloader
            trainset = ESC50(
                wav_dir=os.path.join(dataset_dir, 'audio'),
                csv_path=os.path.join(dataset_dir, 'meta', 'esc50.csv'),
                fold=train_fold
            )
            trainsampler = SimpleSampler(trainset, batch_size, shuffle=True, drop_last=False, fix_class=slt_cls)
            evaluator = Evaluator(cfg, eval_fold=eval_fold, slt_cls=slt_cls)
        elif cfg['DATASOURCE']['NAME'] == 'fsd50k':
            trainset = FSD50K(
                audio_dir=os.path.join(dataset_dir, 'dev_audio'),
                csv_path=os.path.join(dataset_dir, 'ground_truth', 'dev.csv'),
                vocabulary_path=os.path.join(dataset_dir, 'ground_truth', 'vocabulary.csv'),
                clips_dir=os.path.join(cfg['DATASOURCE']['CLIPS_DIR']),
                mode='train',
                num_class=cfg['DATASOURCE']['NUM_CLASS'],
                sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'],
                overwrite_clips=cfg['DATASOURCE']['OVERWRITE_CLIPS']
            )
            trainsampler = SimpleSampler(trainset, batch_size=batch_size, shuffle=True, drop_last=False, fix_class=None)
            evaluator = Evaluator(cfg)
        trainloader = DataLoader(trainset, batch_sampler=trainsampler, num_workers=4, pin_memory=True)
        # Build a extractor
        model = Vgg8(num_class=num_class,
                     sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'],
                     n_fft=cfg['FEATURE_EXTRACTOR']['N_FFT'],
                     win_length=cfg['FEATURE_EXTRACTOR']['WIN_LENGTH'],
                     hop_length=cfg['FEATURE_EXTRACTOR']['HOP_LENGTH'],
                     f_min=cfg['FEATURE_EXTRACTOR']['F_MIN'],
                     f_max=cfg['FEATURE_EXTRACTOR']['F_MAX'],
                     n_mels=cfg['FEATURE_EXTRACTOR']['N_MELS'],
                     window_type=cfg['FEATURE_EXTRACTOR']['WINDOW_TYPE'],
                     include_top=True
                     ).to(device)
        # Resume training if ckpt of pretrained model is provided
        if cfg['MODEL']['PRETRAINED_PATH']:
            ckpt = torch.load(cfg['MODEL']['PRETRAINED_PATH'])
            model.load_state_dict(ckpt)
            logger.info("Load the weight from {cfg['MODEL']['PRETRAINED_PATH']}")

        criterion = loss_fn(name=cfg['TRAINER']['LOSS_FN'], single_target=cfg['DATASOURCE']['SINGLE_TARGET'], reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_performance = 0
        for epoch in range(epochs):
            # Train process
            model.train()

            running_loss = 0
            with tqdm.tqdm(total=len(trainsampler), desc=f"epoch {epoch}/{epochs}", leave=False) as t:
                for i, (batch_x, batch_y) in enumerate(trainloader):
                    # zero the param grads
                    optimizer.zero_grad()

                    batch_outputs = model(batch_x.to(torch.float32).to(device))
                    logger.debug(batch_outputs)
                    logger.debug(batch_y)
                    loss = criterion(batch_outputs, torch.as_tensor(batch_y).to(device))
                    loss.backward()

                    optimizer.step()

                    running_loss += loss.item()  # summarise the loss over batches per epochs

                    # calculate the training results per epoch
                    batch_train_loss = running_loss / (i + 1)
                    # wandb.log({'train_loss': batch_train_loss})
                    # wandb.watch(model)
                    t.set_postfix(training_loss=f"{batch_train_loss:.4f}")
                    t.update()
            logger.info(f"Epoch {epoch+1}: \n"
                        f"training_loss: {running_loss/len(trainloader)}")
            # Evaluate the model
            statistics = evaluator.evaluate(model)
            # Log the metrics to wandb
            # wandb.log(statistics)
            # Add to history values
            for k in statistics.keys():
                if k not in history[fold - 1].keys():
                    history[fold - 1][k] = [statistics[k]]
                else:
                    history[fold - 1][k].append(statistics[k])  # history[0...K-1] corresponds fold 1...K
            # Save the ckpt if achieve the best performance
            current_performance = statistics['acc'] if cfg['DATASOURCE']['SINGLE_TARGET'] else statistics[
                'map']
            if ckpt_dir and (current_performance > best_performance):
                make_folder(ckpt_dir)
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f"epoch_{epoch}-{current_performance:.4f}.pth"))
                best_performance = current_performance
    # Calculate macro result over multiple fold
    result = dict()
    for fold_h in history:
        for key in fold_h.keys():
            if key in result.keys():
                result[key] += np.amax(fold_h[key])
            else:
                result[key] = np.amax(fold_h[key])

    logger.info(f"Summarise this trained model's performance:")
    for key in result.keys():
        result[key] /= K  # average the result of different folds
        logger.info(f"Overall_eval_{key}={result[key]:.6f}")


def eval(cfg: OmegaConf) -> None:
    """ Evaluate the trained model"""
    num_class = cfg['DATASOURCE']['NUM_CLASS']
    if cfg['DATASOURCE']['LABEL_SPLIT']:
        # todo: select different split methods
        nvl_cls = [5, 0, 4, 18, 13, 19, 29, 21, 28, 33, 38, 31, 47, 46, 44]
        logger.info(f"Novel split contains {nvl_cls}")
        slt_cls = [x for x in range(num_class) if x not in nvl_cls]
    else:
        slt_cls = [x for x in range(num_class)]
    logger.info(f"Now evaluating on {slt_cls}")

    for fold in range(1, cfg['DATASOURCE']['K'] + 1):
        logger.info('====================================================================================')
        logger.info(f"Experiments: {fold}/{K}")
        if cfg['DATASOURCE']['NAME'] == 'esc50':
            eval_fold = [fold]
            evaluator = Evaluator(cfg, eval_fold=eval_fold, slt_cls=slt_cls)
        elif cfg['DATASOURCE']['NAME'] == 'fsd50k':
            evaluator = Evaluator(cfg)

    model = Vgg8(num_class=num_class,
                 sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'],
                 n_fft=cfg['FEATURE_EXTRACTOR']['N_FFT'],
                 win_length=cfg['FEATURE_EXTRACTOR']['WIN_LENGTH'],
                 hop_length=cfg['FEATURE_EXTRACTOR']['HOP_LENGTH'],
                 f_min=cfg['FEATURE_EXTRACTOR']['F_MIN'],
                 f_max=cfg['FEATURE_EXTRACTOR']['F_MAX'],
                 n_mels=cfg['FEATURE_EXTRACTOR']['N_MELS'],
                 window_type=cfg['FEATURE_EXTRACTOR']['WINDOW_TYPE'],
                 include_top=True
                 ).to(device)
    if not cfg['MODEL']['PRETRAINED_PATH']:
        logger.warning("Checkpoint of a trained model must be provided.")
    else:
        ckpt = torch.load(cfg['MODEL']['PRETRAINED_PATH'])
        model.load_state_dict(ckpt)
        logger.info("Load the weight from {cfg['MODEL']['PRETRAINED_PATH']}")

    evaluator.evaluate(model)


class Evaluator(object):
    def __init__(self, cfg: OmegaConf, **kwargs):
        # Opts
        dataset_dir = cfg['DATASOURCE']['DATASET_DIR']
        self.single_target = cfg['DATASOURCE']['SINGLE_TARGET']
        self.criterion = loss_fn(name=cfg['TRAINER']['LOSS_FN'], single_target=cfg['DATASOURCE']['SINGLE_TARGET'], reduction='sum')
        self.metrics = Metrics(single_target=cfg['DATASOURCE']['SINGLE_TARGET'])
        self.device = torch.device('cuda') if cfg['TRAINER']['CUDA'] and torch.cuda.is_available() else torch.device('cpu')

        if cfg['DATASOURCE']['NAME'] == 'esc50':
            fold, slt_cls = kwargs['eval_fold'], kwargs['slt_cls']
            dataset = ESC50(wav_dir=os.path.join(dataset_dir, 'audio'),
                            csv_path=os.path.join(dataset_dir, 'meta', 'esc50.csv'),
                            fold=fold)
            datasampler = SimpleSampler(dataset, cfg['TRAINER']['BATCH_SIZE'], shuffle=True, drop_last=False, fix_class=slt_cls)
            self.dataloader = DataLoader(dataset, batch_sampler=datasampler, num_workers=4, pin_memory=True)
            logger.info(f"Now evaluate the model on the classes {slt_cls} with the fold={fold}")

        elif cfg['DATASOURCE']['NAME'] == 'fsd50k':
            dataset = FSD50K(
                audio_dir=os.path.join(dataset_dir, 'dev_audio'),
                csv_path=os.path.join(dataset_dir, 'ground_truth', 'dev.csv'),
                vocabulary_path=os.path.join(dataset_dir, 'ground_truth', 'vocabulary.csv'),
                clips_dir=os.path.join(cfg['DATASOURCE']['CLIPS_DIR']),
                mode='val',
                num_class=cfg['DATASOURCE']['NUM_CLASS'],
                sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'],
                overwrite_clips=cfg['DATASOURCE']['OVERWRITE_CLIPS']
            )
            datasampler = SimpleSampler(dataset, cfg['TRAINER']['BATCH_SIZE'], shuffle=True, drop_last=False, fix_class=None)
            self.dataloader = DataLoader(dataset, batch_sampler=datasampler, num_workers=4, pin_memory=True)


    def evaluate(self,
                 model: torch.nn.Module) -> dict:
        """ Evaluate the model's performance."""
        model.eval()

        running_loss = 0
        predictions = list()
        labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(self.dataloader)):
                batch_outputs = model(batch_x.to(torch.float32).to(self.device))

                loss = self.criterion(batch_outputs, torch.as_tensor(batch_y).to(self.device))
                running_loss += loss.item()
                if self.single_target:
                    preds = np.argmax(batch_outputs.detach().to('cpu').numpy(),
                                      axis=1).tolist() # convert ont-hot label into label idx and add it to the list
                else:
                    batch_outputs = batch_outputs.sigmoid().detach().to('cpu').numpy()
                    n_samples, _ = batch_outputs.shape
                    preds = [batch_outputs[i, :] for i in range(n_samples)]

                predictions.extend(preds)
                labels.extend(batch_y.detach().to('cpu').numpy())
        logger.debug(predictions)
        statistics = self.metrics.summary(labels, predictions)
        statistics['loss'] = running_loss / len(self.dataloader)
        for k, v in statistics.items():
            logger.info(f"val_{k}={v}")
        return statistics


@hydra.main(config_path=config_path, config_name=config_name.split('/')[0])
def main(cfg: OmegaConf) -> None:
    if cfg['MODE'] == 'train':
        train(cfg)
    elif cfg['MODE'] == 'eval':
        eval(cfg)


if __name__ == '__main__':
    main()