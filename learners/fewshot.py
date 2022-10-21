import os
import sys
import time
import tqdm
import yaml
import hydra
# import wandb
import torch
import numpy as np
from string import Template
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from typing import Optional

sys.path.insert(0, '../')
import src.label_ops as lops
from src.fsd_meta import fsd50k_select_ids, fsd50k_blacklist
from src.manifolds import make_folder, set_logger
from pytorch.models import Vgg8, MatchingNetwork
from pytorch.utils import collate_data, loss_fn, task_label, prepare_fewshot_task
from pytorch.eval_kit import Metrics
from pytorch.data import (
    ESC50, ESC50FewShotSampler, FSD50K, MultiLabelFewShotSampler, One2RestSampler, HierOne2RestSampler,
    TaskDependentSampler
)
from pytorch.fewshot import (
    Protonet_onEpisode, Matchnet_onEpisode, HierProtonet_onEpisode, TaxonomyBased_onEpisode, MAML_onEpisode,
    TaDProtonet_onEpisode
)

# Automatically allocate a spare gpu
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
os.system('rm tmp')

# Set configuration file
config_path = '../cfg'
config_name = 'fewshot.yaml'
# Monitor the model performance, see more in https://github.com/wandb/client
with open(os.path.join(config_path, config_name)) as f:
    yaml_data = yaml.safe_load(f)
t = Template(yaml_data['OUTPUTS']['DIR'].replace('.', '__'))
# todo
output_dir = t.substitute(
    DATASOURCE__NAME=yaml_data['defaults'][0]['DATASOURCE'],
    ALGORITHM__NAME=yaml_data['defaults'][2]['ALGORITHM'],
    FEWSHOT_SET__DATASAMPLING__train=yaml_data['FEWSHOT_SET']['DATASAMPLING']['train'],
    TRAINER__LEARNING_RATE=float(yaml_data['TRAINER']['LEARNING_RATE'])
)
output_dir = os.path.join(output_dir, f"{time.strftime('%b-%d_%H-%M', time.localtime())}")
# os.environ['WANDB_DIR'] = output_dir
# make_folder(os.environ['WANDB_DIR'])
# wandb.init(project="AudioTagging", entity="jinhua_liang")
# Multiple processing
torch.multiprocessing.set_sharing_strategy('file_system')
# Set root logger
logger = set_logger(log_name='fewshot', log_dir=output_dir)


def train(cfg: OmegaConf) -> None:
    """ Train a specific model"""
    # Define local dir, param, and func
    dataset_dir = cfg['DATASOURCE']['DATASET_DIR']
    n_novel_cls = cfg['FEWSHOT_SET']['NUM_NVL_CLS']
    n_support_per_cls = cfg['FEWSHOT_SET']['NUM_SUPPORT_PER_CLS']
    n_query_per_cls = cfg['FEWSHOT_SET']['NUM_QUERY_PER_CLS']
    device = torch.device('cuda') if cfg['FEWSHOT_SET']['CUDA'] and torch.cuda.is_available() else torch.device('cpu')
    # Cross-validation
    history = [{} for _ in range(cfg['TRAINER']['K'])]  # k-folds cross validation
    for fold in range(cfg['TRAINER']['K']):  # K is defined arbitrarily as we can constitute different splits
        logger.info('====================================================================================')
        logger.info(f"Experiments: {fold + 1}/{cfg['TRAINER']['K']}")
        # Set dataset & evaluator
        if cfg['DATASOURCE']['NAME'] == 'esc50':
            csv_path = os.path.join(dataset_dir, 'meta', 'esc50.csv')
            all_fold = [x for x in range(1, 6)]  # use all the fold per class

            trainset = ESC50(
                wav_dir=os.path.join(dataset_dir, 'audio'),
                csv_path=csv_path,
                fold=all_fold,
                num_class=cfg['DATASOURCE']['NUM_CLASS'],
                sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE']
            )
            # Set up data split for few shot learning & evaluation
            novel_splt_size = int(cfg['DATASOURCE']['NUM_CLASS'] * cfg['FEWSHOT_SET']['LABEL_SPLIT'][0])
            if cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'random':
                base_split, novel_split = lops.random_split(list(range(cfg['DATASOURCE']['NUM_CLASS'])),
                                                            n_novel_cls=novel_splt_size)
            elif cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'uniform':
                base_split, novel_split = lops.uniform_split(csv_path=csv_path, n_novel_cls=novel_splt_size)
            elif cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'parent':
                base_split, novel_split = lops.parent_split(csv_path=csv_path, prnt_id=(fold + 1),
                                                            n_novel_cls=novel_splt_size)
            elif cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'select':
                assert len(lops.esc50_select_ids) == cfg['TRAINER']['K']
                novel_split = lops.esc50_select_ids[fold]
                base_split = [x for x in range(cfg['DATASOURCE']['NUM_CLASS']) if x not in novel_split]
                if len(novel_split) != novel_splt_size:
                    logger.warning(
                        f"Mismatched split size: select {len(novel_splt_size)} labels instead of {novel_splt_size}")
                assert (set(base_split) & set(novel_split) == set())
            evaluator = Evaluator(cfg, eval_fold=all_fold, slt_cls=novel_split)

        elif cfg['DATASOURCE']['NAME'] == 'fsd50k':
            audio_dirs = [os.path.join(dataset_dir, 'dev_audio'), os.path.join(dataset_dir, 'eval_audio')]
            csv_paths = [os.path.join(dataset_dir, 'ground_truth', 'dev.csv'),
                         os.path.join(dataset_dir, 'ground_truth', 'eval.csv')]
            trainset = FSD50K(
                audio_dir=audio_dirs,
                csv_path=csv_paths,
                vocabulary_path=os.path.join(dataset_dir, 'ground_truth', 'vocabulary.csv'),
                clips_dir=cfg['DATASOURCE']['CLIPS_DIR'],
                mode='fewshot',
                num_class=cfg['DATASOURCE']['NUM_CLASS'],
                sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'],
                overwrite_clips=cfg['DATASOURCE']['OVERWRITE_CLIPS']
            )

            # Set up data split for few shot learning & evaluation
            if cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'random':
                base_split, novel_split = lops.random_split(list(range(cfg['DATASOURCE']['NUM_CLASS'])),
                                                            n_novel_cls=n_novel_cls)
                evaluator = Evaluator(cfg, slt_cls=novel_split)
            elif cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'select':
                base_split = lops.fsd50k_select_ids['base']
                nvl_val_split = lops.fsd50k_select_ids['novel_val']
                nvl_eval_split = lops.fsd50k_select_ids['novel_eval']
                assert len(set(base_split) & set(nvl_val_split)) == 0
                assert len(set(base_split) & set(nvl_eval_split)) == 0
                assert len(set(nvl_val_split) & set(nvl_eval_split)) == 0
                base_split = base_split
                novel_split = nvl_val_split
                block = base_split + fsd50k_blacklist
                evaluator = Evaluator(cfg, slt_cls=novel_split, blacklist=block)
        else:
            logger.warning(f"Cannot recognise `dataset` {cfg['DATASOURCE']['NAME']}")

        # Set data sampler
        if cfg['FEWSHOT_SET']['DATASAMPLING']['train'] == 'esc50_fs':
            trainsampler = ESC50FewShotSampler(
                dataset=trainset,
                num_nvl_cls=n_novel_cls,
                num_sample_per_cls=n_support_per_cls,
                num_queries_per_cls=n_query_per_cls,
                num_task=cfg['FEWSHOT_SET']['NUM_TASK'],
                batch_size=cfg['ALGORITHM']['BATCH_SIZE'],
                fix_split=base_split,
                require_pid=cfg['TRAINER']['REQUIRE_PID']
            )
        elif cfg['FEWSHOT_SET']['DATASAMPLING']['train'] == 'multilabel_fs':
            trainsampler = MultiLabelFewShotSampler(
                dataset=trainset,
                n_nvl_cls=n_novel_cls,
                n_support_per_cls=n_support_per_cls,
                n_query_per_cls=n_query_per_cls,
                n_task=cfg['FEWSHOT_SET']['NUM_TASK'],
                batch_size=cfg['ALGORITHM']['BATCH_SIZE'],
                fix_split=base_split,
                blacklist=fsd50k_blacklist
            )
        elif cfg['FEWSHOT_SET']['DATASAMPLING']['train'] == 'one_vs_rest':
            trainsampler = One2RestSampler(
                dataset=trainset,
                n_nvl_cls=n_novel_cls,
                n_supports_per_cls=n_support_per_cls,
                n_task=-1,
                fix_split=base_split,
                blacklist=fsd50k_blacklist,
                is_shuffle=True
            )
            n_query_per_cls = 1 / n_novel_cls
        elif cfg['FEWSHOT_SET']['DATASAMPLING']['train'] == 'hier_one_vs_rest':
            trainsampler = HierOne2RestSampler(
                taxonomy_path=cfg['DATASOURCE']['TAXONOMY_PATH'],
                vocabulary_path=os.path.join(dataset_dir, 'ground_truth/vocabulary.csv'),
                dataset=trainset,
                n_nvl_cls=n_novel_cls,
                n_supports_per_cls=n_support_per_cls,
                n_task=-1,
                fix_split=base_split,
                blacklist=fsd50k_blacklist,
                is_shuffle=True
            )
            n_query_per_cls = 1 / n_novel_cls
        elif cfg['FEWSHOT_SET']['DATASAMPLING']['train'] == 'task_dependent':
            assert cfg['ALGORITHM']['NAME'] == 'tad_proto'
            trainsampler = TaskDependentSampler(
                taxonomy_path=cfg['DATASOURCE']['TAXONOMY_PATH'],
                vocabulary_path=os.path.join(dataset_dir, 'ground_truth/vocabulary.csv'),
                label_weights_path=cfg['ALGORITHM']['EMBED_LABEL']['WEIGHTS_PATH'],
                dataset=trainset,
                n_nvl_cls=n_novel_cls,
                n_supports_per_cls=n_support_per_cls,
                n_task=-1,
                fix_split=base_split,
                blacklist=fsd50k_blacklist,
                is_shuffle=True,
                is_embed_label=cfg['ALGORITHM']['EMBED_LABEL']['STATUS'],
                beta=cfg['ALGORITHM']['EMBED_LABEL']['BETA'],
                is_aligned=cfg['ALGORITHM']['ALIGN_TASK']
            )
            n_query_per_cls = 1 / n_novel_cls
        else:
            logger.warning(f"{cfg['FEWSHOT_SET']['DATASAMPLING']['train']} is not in the pool of sampler.")

        trainloader = DataLoader(trainset, batch_sampler=trainsampler, collate_fn=collate_data, num_workers=4,
                                 pin_memory=True)

        if cfg['ALGORITHM']['MODEL']['NAME'] == 'vgg8':
            model = Vgg8(
                num_class=n_novel_cls,
                sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'],
                n_fft=cfg['FEATURE_EXTRACTOR']['N_FFT'],
                win_length=cfg['FEATURE_EXTRACTOR']['WIN_LENGTH'],
                hop_length=cfg['FEATURE_EXTRACTOR']['HOP_LENGTH'],
                f_min=cfg['FEATURE_EXTRACTOR']['F_MIN'],
                f_max=cfg['FEATURE_EXTRACTOR']['F_MAX'],
                n_mels=cfg['FEATURE_EXTRACTOR']['N_MELS'],
                window_type=cfg['FEATURE_EXTRACTOR']['WINDOW_TYPE'],
                include_top=False
            ).to(device)
        elif cfg['ALGORITHM']['MODEL']['NAME'] == 'match_net':
            model = MatchingNetwork(
                lstm_layers=1,
                lstm_input_size=cfg['ALGORITHM']['MODEL']['INPUT_SIZE'],
                unrolling_steps=cfg['ALGORITHM']['MODEL']['NUM_LSTM'],
                device=device,
                mono=cfg['DATASOURCE']['SINGLE_TARGET']
            ).to(device)
        else:
            raise ValueError(f"Cannot recognise the model 'cfg['ALGORITHM']['MODEL']['NAME']'.")
        # Set loss & optimiser for back propagation in the training
        criterion = loss_fn(name=cfg['ALGORITHM']['LOSS_FN'],
                            single_target=cfg['DATASOURCE']['SINGLE_TARGET'],
                            reduction='sum')
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg['TRAINER']['LEARNING_RATE'])
        # Set on-episode func for few shot learning
        if cfg['ALGORITHM']['NAME'] == 'proto':
            train_on_episode = Protonet_onEpisode(
                n_nvl_cls=n_novel_cls,
                n_support_per_cls=n_support_per_cls,
                n_query_per_cls=n_query_per_cls,
                distance='l2_distance',
                is_mono=cfg['DATASOURCE']['SINGLE_TARGET'],  # True
                criterion=criterion,
                is_train=True,
                optimiser=optimiser,
                require_y=False
            )
        elif cfg['ALGORITHM']['NAME'] == 'match':
            train_on_episode = Matchnet_onEpisode(
                n_nvl_cls=n_novel_cls,
                n_support_per_cls=n_support_per_cls,
                n_query_per_cls=n_query_per_cls,
                distance='l2_distance',  # 'cosine'
                is_mono=cfg['DATASOURCE']['SINGLE_TARGET'],
                criterion=criterion,
                is_train=True,
                optimiser=optimiser,
                require_y=False
            )
        elif cfg['ALGORITHM']['NAME'] == 'maml':
            train_on_episode = MAML_onEpisode(
                n_nvl_cls=n_novel_cls,
                n_support_per_cls=n_support_per_cls,
                n_query_per_cls=n_query_per_cls,
                distance='l2_distance',
                is_mono=True,
                is_train=True,
                is_approximate=True,
                criterion=criterion,
                optimiser=optimiser,
                base_tr_step=5,
                base_lr=1e-3,
                require_y=False
            )
        elif cfg['ALGORITHM']['NAME'] == 'hier_proto':
            train_on_episode = HierProtonet_onEpisode(
                n_nvl_cls=n_novel_cls,
                n_support_per_cls=n_support_per_cls,
                n_query_per_cls=n_query_per_cls,
                height=2,
                alpha=-1,
                distance='l2_distance',
                criterion=criterion,
                is_train=True,
                optimiser=optimiser,
                require_y=False
            )
        elif cfg['ALGORITHM']['NAME'] == 'tad_proto':
            train_on_episode = TaDProtonet_onEpisode(
                n_nvl_cls=n_novel_cls,
                n_support_per_cls=n_support_per_cls,
                n_query_per_cls=n_query_per_cls,
                is_aligned=cfg['ALGORITHM']['ALIGN_TASK'],
                distance='l2_distance',
                is_mono=cfg['DATASOURCE']['SINGLE_TARGET'],
                criterion=criterion,
                is_train=True,
                optimiser=optimiser,
                require_y=False
            )
        elif cfg['ALGORITHM']['NAME'] == 'tab':
            train_on_episode = TaxonomyBased_onEpisode(
                n_nvl_cls=n_novel_cls,
                n_support_per_cls=n_support_per_cls,
                n_query_per_cls=n_query_per_cls,
                alpha=0,
                distance='l2_distance',
                criterion=criterion,
                is_train=True,
                optimiser=optimiser,
                require_y=False
            )
        # Resume training if ckpt of pretrained model is provided
        if cfg['TRAINER']['RESUME_TRAINING']:
            ckpt = torch.load(cfg['ALGORITHM']['MODEL']['PRETRAINED_PATH'])
            model.load_state_dict(ckpt)
            logger.info("Load the weight from {cfg['ALGORITHM']['MODEL']['PRETRAINED_PATH']}")

        """ Training on episodes."""
        best_score = 0
        for epoch in range(cfg['TRAINER']['EPOCHS']):
            model.train()

            running_loss = 0
            with tqdm.tqdm(total=len(trainloader), desc=f"epoch {epoch}/{cfg['TRAINER']['EPOCHS']}") as t:
                for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(trainloader)):
                    loss = train_on_episode(model, batch_x.to(device), batch_y.to(device))
                    running_loss += loss
                    # wandb.log({'train_loss': (running_loss / (i + 1))})
                    # wandb.watch(model)
                    t.set_postfix(loss=f"{(running_loss / (i + 1)):.4f}")
                    t.update()
                    # if i == 1:
                    #     break
            logger.info(f"Epoch {epoch}: train_loss={(running_loss / len(trainloader))}")
            # Evaluate the model
            statistics = evaluator.evaluate(model)
            # Log the metrics to wandb
            # wandb.log(statistics)
            # Add to history values
            for k in statistics.keys():
                if k not in history[fold].keys():
                    history[fold][k] = [statistics[k]]
                else:
                    history[fold][k].append(statistics[k])
            # Save the checkpoint when conducting the best performance
            current_score = statistics['acc'] if cfg['DATASOURCE']['SINGLE_TARGET'] else statistics['map']
            if output_dir and (current_score > best_score):
                ckpt_dir = os.path.join(output_dir, 'ckpts')
                make_folder(ckpt_dir)
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f"epoch_{epoch}-{current_score:.4f}.pth"))
                best_score = current_score
            logger.info(f"epoch {epoch}: the best performance of training model is {best_score}")

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
        result[key] /= cfg['TRAINER']['K']  # average the result of different folds
        logger.info(f"Overall_eval_{key}={result[key]:.6f}")


def test(cfg: OmegaConf, novel_cls: Optional[list] = None) -> None:
    """ Evaluate the trained model."""
    # Define local dir, param, and func
    dataset_dir = cfg['DATASOURCE']['DATASET_DIR']
    n_novel_cls = cfg['FEWSHOT_SET']['NUM_NVL_CLS']
    device = torch.device('cuda') if cfg['FEWSHOT_SET']['CUDA'] and torch.cuda.is_available() else torch.device('cpu')
    # Modify some opts in cfg
    cfg['FEWSHOT_SET']['NUM_QUERY_PER_CLS'] = cfg['TESTER']['NUM_QUERY_PER_CLS']
    cfg['FEWSHOT_SET']['SHUFFLE_CLASS'] = cfg['TESTER']['SHUFFLE_CLASS']
    # Set up data split for evaluation
    if cfg['DATASOURCE']['NAME'] == 'esc50':
        csv_path = os.path.join(dataset_dir, 'meta', 'esc50.csv')
        all_fold = [x for x in range(1, 6)]  # use all the fold per class
        fold_id = 1  # fold id reserved for test [1, 6)

        novel_splt_size = int(cfg['DATASOURCE']['NUM_CLASS'] * cfg['FEWSHOT_SET']['LABEL_SPLIT'][0])
        if cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'random':
            _, novel_cls = lops.random_split(list(range(cfg['DATASOURCE']['NUM_CLASS'])),
                                             n_novel_cls=novel_splt_size)
        elif cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'uniform':
            _, novel_cls = lops.uniform_split(csv_path=csv_path, n_novel_cls=novel_splt_size)
        elif cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'parent':
            _, novel_cls = lops.parent_split(csv_path=csv_path, prnt_id=fold_id, n_novel_cls=novel_splt_size)
        elif cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'select':
            if len(novel_cls) != novel_splt_size:
                logger.warning(
                    f"Mismatched split size: select {len(novel_splt_size)} labels instead of {novel_splt_size}")
        logger.info(f"Now testing on {novel_cls}")
        evaluator = Evaluator(cfg, eval_fold=all_fold, slt_cls=novel_cls)

    elif cfg['DATASOURCE']['NAME'] == 'fsd50k':
        if cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'random':
            _, novel_split = lops.random_split(list(range(cfg['DATASOURCE']['NUM_CLASS'])), n_novel_cls=n_novel_cls)
            evaluator = Evaluator(cfg, slt_cls=novel_split)
        elif cfg['FEWSHOT_SET']['LABEL_SPLIT'][1] == 'select':
            base_split = lops.fsd50k_select_ids['base']
            novel_split = lops.fsd50k_select_ids['novel_eval']
            assert len(set(base_split) & set(novel_split)) == 0
            block = base_split + fsd50k_blacklist
            evaluator = Evaluator(cfg, slt_cls=novel_split, blacklist=block)
    else:
        logger.warning(f"Cannot recognise `dataset` {cfg['DATASOURCE']['NAME']}")

    # Set audio encoder
    if cfg['ALGORITHM']['MODEL']['NAME'] == 'vgg8':
        model = Vgg8(
            num_class=n_novel_cls,
            sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'],
            n_fft=cfg['FEATURE_EXTRACTOR']['N_FFT'],
            win_length=cfg['FEATURE_EXTRACTOR']['WIN_LENGTH'],
            hop_length=cfg['FEATURE_EXTRACTOR']['HOP_LENGTH'],
            f_min=cfg['FEATURE_EXTRACTOR']['F_MIN'],
            f_max=cfg['FEATURE_EXTRACTOR']['F_MAX'],
            n_mels=cfg['FEATURE_EXTRACTOR']['N_MELS'],
            window_type=cfg['FEATURE_EXTRACTOR']['WINDOW_TYPE'],
            include_top=False
        ).to(device)
    elif cfg['ALGORITHM']['MODEL']['NAME'] == 'match_net':
        model = MatchingNetwork(
            lstm_layers=1,
            lstm_input_size=cfg['ALGORITHM']['MODEL']['INPUT_SIZE'],
            unrolling_steps=cfg['ALGORITHM']['MODEL']['NUM_LSTM'],
            device=device,
            mono=cfg['DATASOURCE']['SINGLE_TARGET']
        ).to(device)
    else:
        raise ValueError(f"Cannot recognise the model 'cfg['ALGORITHM']['MODEL']['NAME']'.")

    # Load weights of trained model
    if not cfg['ALGORITHM']['MODEL']['PRETRAINED_PATH']:
        logger.warning("Checkpoint of a trained model must be provided.")
    else:
        ckpt = torch.load(cfg['ALGORITHM']['MODEL']['PRETRAINED_PATH'])
        model.load_state_dict(ckpt)
        logger.info(f"Load the weight from '{cfg['ALGORITHM']['MODEL']['PRETRAINED_PATH']}'.")

    statistics = evaluator.evaluate(model)
    for k in statistics.keys():
        _res = np.mean(statistics[k])
        logger.info(f"Overall_eval_{k}={_res:.6f}")


class Evaluator(object):
    def __init__(self, cfg: OmegaConf, **kwargs):
        # Opts
        dataset_dir = cfg['DATASOURCE']['DATASET_DIR']
        mode = cfg['FEWSHOT_SET']['MODE']
        self.single_target = cfg['DATASOURCE']['SINGLE_TARGET']
        self.criterion = loss_fn(name=cfg['ALGORITHM']['LOSS_FN'], single_target=cfg['DATASOURCE']['SINGLE_TARGET'],
                                 reduction='sum')
        avg = None if cfg['FEWSHOT_SET']['MODE'] == 'test' else 'macro'
        self.metrics = Metrics(single_target=cfg['DATASOURCE']['SINGLE_TARGET'], average=avg)
        self.device = torch.device('cuda') if cfg['FEWSHOT_SET']['CUDA'] and torch.cuda.is_available() else torch.device(
            'cpu')
        # Few shot params
        self.n_novel_cls = cfg['FEWSHOT_SET']['NUM_NVL_CLS']
        self.n_support_per_cls = cfg['FEWSHOT_SET']['NUM_SUPPORT_PER_CLS']
        self.n_query_per_cls = cfg['FEWSHOT_SET']['NUM_QUERY_PER_CLS']
        slt_cls = kwargs['slt_cls']
        # Set data source
        if cfg['DATASOURCE']['NAME'] == 'esc50':
            fold = kwargs['eval_fold']
            dataset = ESC50(wav_dir=os.path.join(dataset_dir, 'audio'),
                            csv_path=os.path.join(dataset_dir, 'meta', 'esc50.csv'),
                            fold=fold,
                            num_class=cfg['DATASOURCE']['NUM_CLASS'],
                            sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'])
            logger.info(f"Now evaluate the model on the classes {slt_cls} with the fold={fold}")
        elif cfg['DATASOURCE']['NAME'] == 'fsd50k':
            audio_dirs = [os.path.join(dataset_dir, 'dev_audio'), os.path.join(dataset_dir, 'eval_audio')]
            csv_paths = [os.path.join(dataset_dir, 'ground_truth', 'dev.csv'),
                         os.path.join(dataset_dir, 'ground_truth', 'eval.csv')]
            dataset = FSD50K(
                audio_dir=audio_dirs,
                csv_path=csv_paths,
                vocabulary_path=os.path.join(dataset_dir, 'ground_truth', 'vocabulary.csv'),
                clips_dir=os.path.join(cfg['DATASOURCE']['CLIPS_DIR']),
                mode='fewshot',
                num_class=cfg['DATASOURCE']['NUM_CLASS'],
                sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'],
                overwrite_clips=False
            )
        # Set data sampling
        if cfg['FEWSHOT_SET']['DATASAMPLING']['eval'] == 'esc50_fs':
            datasampler = ESC50FewShotSampler(
                dataset=dataset,
                num_nvl_cls=self.n_novel_cls,
                num_sample_per_cls=self.n_support_per_cls,
                num_queries_per_cls=self.n_query_per_cls,
                num_task=cfg['FEWSHOT_SET']['NUM_TASK'],
                batch_size=cfg['ALGORITHM']['BATCH_SIZE'],
                fix_split=slt_cls,
                require_pid=cfg['TRAINER']['REQUIRE_PID']
            )
        elif cfg['FEWSHOT_SET']['DATASAMPLING']['eval'] == 'multilabel_fs':
            datasampler = MultiLabelFewShotSampler(
                dataset=dataset,
                n_nvl_cls=self.n_novel_cls,
                n_support_per_cls=self.n_support_per_cls,
                n_query_per_cls=self.n_query_per_cls,
                n_task=cfg['FEWSHOT_SET']['NUM_TASK'],
                batch_size=cfg['ALGORITHM']['BATCH_SIZE'],
                fix_split=slt_cls,
                blacklist=kwargs['blacklist'],
                shuffle_class=cfg['FEWSHOT_SET']['SHUFFLE_CLASS']
            )
        else:
            logger.warning(f"{cfg['FEWSHOT_SET']['DATASAMPLING']['eval']} is not in the pool of sampler.")
        self.dataloader = DataLoader(dataset, batch_sampler=datasampler, collate_fn=collate_data, num_workers=4,
                                     pin_memory=True)
        # Set on-episode func for few shot learning
        if cfg['ALGORITHM']['NAME'] == 'proto':
            self.eval_on_episode = Protonet_onEpisode(
                n_nvl_cls=self.n_novel_cls,
                n_support_per_cls=self.n_support_per_cls,
                n_query_per_cls=self.n_query_per_cls,
                distance='l2_distance',
                is_mono=self.single_target,
                criterion=self.criterion,
                is_train=False,
                optimiser=None,
                require_y=True
            )
        elif cfg['ALGORITHM']['NAME'] == 'match':
            self.eval_on_episode = Matchnet_onEpisode(
                n_nvl_cls=self.n_novel_cls,
                n_support_per_cls=self.n_support_per_cls,
                n_query_per_cls=self.n_query_per_cls,
                distance='l2_distance',  # 'cosine',
                is_mono=self.single_target,
                criterion=self.criterion,
                is_train=False,
                optimiser=None,
                require_y=True
            )
        elif cfg['ALGORITHM']['NAME'] == 'maml':
            self.eval_on_episode = MAML_onEpisode(
                n_nvl_cls=self.n_novel_cls,
                n_support_per_cls=self.n_support_per_cls,
                n_query_per_cls=self.n_query_per_cls,
                distance='l2_distance',
                is_mono=self.single_target,
                is_train=False,
                is_approximate=True,
                criterion=self.criterion,
                optimiser=None,
                base_tr_step=5,
                base_lr=2e-3,
                require_y=True
            )
        elif cfg['ALGORITHM']['NAME'] == 'hier_proto':
            self.eval_on_episode = HierProtonet_onEpisode(
                n_nvl_cls=self.n_novel_cls,
                n_support_per_cls=self.n_support_per_cls,
                n_query_per_cls=self.n_query_per_cls,
                height=2,
                alpha=-1,
                distance='l2_distance',
                criterion=self.criterion,
                is_train=False,
                optimiser=None,
                require_y=True
            )
        elif cfg['ALGORITHM']['NAME'] == 'tad_proto':
            self.eval_on_episode = TaDProtonet_onEpisode(
                n_nvl_cls=self.n_novel_cls,
                n_support_per_cls=self.n_support_per_cls,
                n_query_per_cls=self.n_query_per_cls,
                is_aligned=cfg['ALGORITHM']['ALIGN_TASK'],
                distance='l2_distance',
                is_mono=self.single_target,
                criterion=self.criterion,
                is_train=False,
                optimiser=None,
                require_y=True
            )
        elif cfg['ALGORITHM']['NAME'] == 'tab':
            self.eval_on_episode = TaxonomyBased_onEpisode(
                n_nvl_cls=self.n_novel_cls,
                n_support_per_cls=self.n_support_per_cls,
                n_query_per_cls=self.n_query_per_cls,
                alpha=0,
                distance='l2_distance',
                criterion=self.criterion,
                is_train=False,
                optimiser=None,
                require_y=True
            )
        elif cfg['ALGORITHM']['NAME'] == 'lr':
            self.eval_on_episode = LinearRegression(
                n_nvl_cls=self.n_novel_cls,
                n_support_per_cls=self.n_support_per_cls,
                n_query_per_cls=self.n_query_per_cls,
                is_mono=self.single_target,
                criterion=self.criterion,
                optimiser=None,
                freeze_encoder=True,
                base_tr_step=5,
                require_y=True
            )

    def evaluate(self, model: torch.nn.Module) -> dict:
        """ Evaluate the model's performance."""
        model.eval()

        running_loss = 0
        predictions = list()
        labels = list()
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(self.dataloader)):
                loss, batch_outputs, ground_truth = self.eval_on_episode(model, batch_x.to(self.device),
                                                                         batch_y.to(self.device))
                running_loss += loss
                if self.single_target:
                    preds = np.argmax(batch_outputs.numpy(), axis=1).tolist()  # convert ont-hot label into label idx
                    gt = ground_truth.numpy().tolist()
                else:
                    batch_outputs, ground_truth = batch_outputs.numpy(), ground_truth.numpy()
                    n_samples, _ = batch_outputs.shape
                    assert n_samples == ground_truth.shape[0]
                    preds = [batch_outputs[i, :] for i in range(n_samples)]
                    gt = [ground_truth[i, :] for i in range(n_samples)]
                predictions.extend(preds)
                labels.extend(gt)
        statistics = self.metrics.summary(labels, predictions)
        statistics['loss'] = running_loss / len(self.dataloader)
        for k, v in statistics.items():
            logger.info(f"val_{k}={v}")
        return statistics


@hydra.main(config_path=config_path, config_name=config_name.split('/')[0])
def main(cfg: OmegaConf) -> None:
    logger.info(
        f"====================================================================================\n"
        f"Configuration:"
        f"{cfg}"
    )
    if cfg['FEWSHOT_SET']['MODE'] == 'train':
        train(cfg)
    elif cfg['FEWSHOT_SET']['MODE'] == 'test':
        test(cfg)


if __name__ == '__main__':
    main()
