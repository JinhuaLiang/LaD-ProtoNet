import os
import time
import logging
import numpy as np


def set_logger(level='debug', log_name='default', log_dir=None):
    """ Create logger to record result and output to console and file at the same time."""
    if level == 'debug':
        security = logging.DEBUG
    elif level == 'info':
        security = logging.INFO
    elif level == 'warning':
        security = logging.WARNING
    elif level == 'error':
        security = logging.ERROR
    # Initialise a root logger
    logger = logging.getLogger(name=log_name)
    logger.setLevel(logging.DEBUG) # set logger to the lowest level
    formatter = logging.Formatter(
        fmt='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%b.%d,%Y-%H:%M:%S'
    )
    # Set console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(security)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Set file handler
    if log_dir:
        make_folder(log_dir)
        log_path = os.path.join(log_dir, f"{time.strftime('%b-%d_%H-%M', time.localtime())}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(security)
        logger.addHandler(file_handler)

    logger.info(f"The logger is set already,"
                f"Ones can retrieve the result on the console,"
                f"or by reading the file {log_dir}")
    return logger


def make_folder(path_to_folder):
    """ Create a folder recursively."""
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)


def one_hot(label, num_class):
    """ convert labels to one-hot format
        :param: label: int/float or list
        :param: num_class: int/float
        :returns: np.array     
    """
    def _make_one_hot(label_idx, num_class):
        output = np.zeros(num_class)
        if isinstance(label_idx, str): 
            output[int(label_idx)] = 1 # convert to int from str drawn from csv file
        else:
            output[label_idx] = 1
        return output

    if isinstance(label, list):
        one_hot_labels = []
        for l in label:
            one_hot_l = _make_one_hot(l, num_class)
            one_hot_labels.append(one_hot_l)
        return one_hot_labels
    else:
        return _make_one_hot(one_hot_l, num_class)


def float32_to_int16(data):
    """
    convert data format from float32 to int16
    """
    # float32 format uniforms the data to range [-1, 1],
    # so we need times 2^15 to match the range of int16
    assert np.max(np.abs(data)) <= 1.2
    res = np.clip(data, -1, 1)

    return (res * 32767.).astype(np.int16)


def int16_to_float32(data):
    """
    inverse version of float32_to_int16
    """
    return (data / 32767).astype(np.float32)


def replicate_and_truncate(wav, window_length, overlap):
    clips = []
    if len(wav) < window_length:
        tmp = _replicate(wav, window_length)
        clips.append(tmp)
    else:
        hop_length = int(window_length - overlap)
        for idx in range(0, len(wav), hop_length):
            tmp = wav[idx:idx + window_length]
            tmp = _replicate(tmp, window_length)  # to ensure the last seq have the same length
            clips.append(tmp)
    return clips


def _replicate(x, min_clip_duration):
    """
    replicate clips shorter than 1 second as per the original paper
    :param x: list
    :param min_clip_duration: scalar, the time samples in each Time-Frequency patch
    :return: np.array, 1-D tensor, [min_clip_duration, ]
    """
    if len(x) < min_clip_duration:
        tile_size = (min_clip_duration // x.shape[0]) + 1
        x = np.tile(x, tile_size)[:min_clip_duration]
    return x


def generate_task_label(num_cls: int, num_sample_per_cls: int) -> np.ndarray:
    """ Create structured label for an independent task in few shot learning
        Args:
            num_cls: number of class in few shot learning, i.e., 'n ways'
            num_sample_per_cls: number of samples per class, e.g.,, 'K shots' or number of queries
        E.g.:
            [0]*q + [1]*q + ... + [k-1]*q = generate_task_label(k, q)
    """
    return np.linspace(0, num_cls, num_sample_per_cls*num_cls, endpoint=False).astype(int)