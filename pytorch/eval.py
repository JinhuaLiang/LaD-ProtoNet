import os
import sys
import time
import logging

sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '../utils'))
sys.path.insert(0, os.path.join(sys.path[0], '../cfgs'))

import argparse
import tqdm
import numpy as np

import cfgs_fewshot as cfgs

from torch.utils.data import DataLoader

from evaluation import *
from torch_tools import *
from dataset import ESC50, FewShotSampler
from models import Vgg8
from manifolds import make_folder
from prepare_vocabulary import *
from proto import protonet_on_episode

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

output_dir = cfgs.OUTPUT_DIR
num_task = cfgs.NUM_TASK
num_cls = cfgs.NUM_NVL_CLS
num_support_per_cls = cfgs.NUM_SUPPORT_PER_CLS
num_query_per_cls = cfgs.NUM_QUERY_PER_CLS


def eval(args):
    # params
    batch_size = 1  # args.batch_size
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # dirs and paths
    dataset_dir = cfgs.DATASET_DIR
    output_dir = cfgs.OUTPUT_DIR
    wav_dir = os.path.join(dataset_dir, 'audio')
    ckpt_dir = os.path.join(output_dir, 'ckpts', f"{time.strftime('%b-%d_%H-%M', time.localtime())}")

    # dataset and hyper-params
    csv_path = os.path.join(dataset_dir, 'meta', 'esc50.csv')

    # K-fold cross-valadation over parent categories
    history = {'loss': [], 'acc': [], 'F1-score': []}

    # _, _, vocabulary = create_vocabulary(csv_path)
    # base_split, novel_split = split_by_parent(vocabulary, nvl_parent_id=fold)
    #
    # # randomly remove 6 classes from base split and add them to novel split
    # # so that there would be 16 novel classes and 34 base classes
    # base_ids, novel_ids = random_split(base_split, 6)
    # base_split = base_ids
    # novel_split.extend(novel_ids)

    # slt_cls = [49, 26, 22, 13, 41, 17, 45, 24, 23, 4, 33, 14, 30, 10, 28, 44, 34, 18, 20, 25, 6, 7, 47, 1, 16, 0, 15, 5, 11, 9, 8,
    #  12, 43, 37]
    # # [27, 35, 40, 38, 2, 3, 48, 29, 46, 31, 32, 39, 21, 36, 19, 42]
    # print(slt_cls)
    # # [7, 33, 43, 23, 2, 3, 39, 1, 35, 49, 9, 32, 45, 21, 26, 6, 38, 40, 27, 44, 48, 37, 41, 5, 36, 4, 0, 25, 22, 8, 31, 47, 46, 42]
    # # [14, 19, 17, 15, 16, 12, 10, 18, 11, 13, 30, 28, 34, 29, 20, 24]
    # novel_split = [27, 35, 40, 38, 2, 3, 48, 29, 46, 31, 32, 39, 21, 36, 19, 42]
    novel_split = [14, 19, 17, 15, 16, 12, 10, 18, 11, 13, 30, 28, 34, 29, 20, 24]

    print('====================================================================================')
    print(f"Validatation...")

    evalset = ESC50(wav_dir, csv_path, [str(x) for x in range(1, 6)])
    eval_sampler = FewShotSampler(evalset, num_cls, num_support_per_cls, num_query_per_cls, num_task, batch_size,
                                  novel_split)
    evalloader = DataLoader(evalset, sampler=eval_sampler, num_workers=4, pin_memory=True)

    model = Vgg8(num_cls, include_top=False)

    params_dict = torch.load('parent_0.8713.pth')
    model_dict = {k: v for k, v in params_dict.items() if k in model.state_dict()}
    model.load_state_dict(model_dict)

    model.to(device)
    # Evaluate the model and save if reach the best acc
    statistics = evaluate(model, evalloader, device)
    for key in statistics.keys():
        history[key].append(statistics[key])


def evaluate(model, dataloader, device):
    running_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for i, (x, _) in tqdm.tqdm(enumerate(dataloader)):
            x, y = prepare_fewshot_task(num_cls, num_support_per_cls, num_query_per_cls)(x, device)
            loss, preds = protonet_on_episode(model,
                                              x, y,
                                              num_cls,
                                              num_support_per_cls,
                                              num_query_per_cls,
                                              distance='l2_distance',
                                              train=False)

            running_loss += loss

            preds = np.argmax(preds.numpy(), axis=1).tolist()  # convert ont-hot label into label idx
            predictions.extend(preds)
            labels.extend(y[num_support_per_cls * num_cls:].detach().to('cpu').numpy())

    acc = metrics.accuracy_score(labels, predictions)
    f1_score = metrics.f1_score(labels, predictions, average='macro')
    statistics = {'loss': running_loss / len(dataloader), 'acc': acc, 'F1-score': f1_score}

    logging.info('val_loss={}, val_acc={}, val_F1-score={} \n'.format(statistics['loss'], statistics['acc'],
                                                                      statistics['F1-score']))
    print('val_loss={}, val_acc={}, val_F1-score={} \n'.format(statistics['loss'], statistics['acc'],
                                                               statistics['F1-score']))
    return statistics


if __name__ == '__main__':
    log_dir = os.path.join(output_dir, 'logs', 'fewshot')
    make_folder(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, f"{time.strftime('%b-%d_%H-%M', time.localtime())}"),
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('eval', help='select train mode')
    parser_train.add_argument('-bs', '--batch_size', type=int, default=32)
    parser_train.add_argument('--cuda', action='store_true', default=False, help='enable cuda acceleration')

    args = parser.parse_args()

    if args.mode == 'eval':
        eval(args)