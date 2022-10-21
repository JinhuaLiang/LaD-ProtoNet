import os
import torch
import torchaudio
from torch import Tensor
from typing import Union, Callable, List, Optional, Tuple


def make_task_labels(y: torch.Tensor, *, single_target: bool = True, slt_cls: int = None) -> torch.Tensor:
    """ Interface func to order labels in a independent task"""
    if single_target:
        return task_id(y)
    else:
        if slt_cls == None:
            pass


def task_id(y: torch.Tensor) -> torch.Tensor:
    """ Convert labels to order labels in a independent task, this function support multi-level label convertion.
        E.p.: [3, 1, 2] = task_label([64, 32, 45])."""
    def _make_task_label(label: torch.Tensor) -> torch.Tensor:
        """ Convert labels to ordered labels in a independent task for few shot learning."""
        _, task_label = torch.unique(label, sorted=False, return_inverse=True)
        return task_label

    if y.dim() == 1:
        y = _make_task_label(y)
    else:
        _, n_level = y.size()
        for lvl in range(n_level):
            y[:, lvl] = _make_task_label(y[:, lvl])
    return y


def get_distance(x, y, metric='l2_distance'):
    """ """
    if metric == 'l2_distance':
        return l2_distance(x, y, square=True)

    elif metric == 'cosine':
        return cosine_distance(x, y, eps=1e-8)

    elif metric == 'dot':
        return dot(x, y)

    else:
        raise KeyError("metric should be selected from `l2_distance`, `cosine`, `dot`")


def l2_distance(x, y, square=True):
    """ Calculate l2 distance between x and y, following 'sub-power^2-sum'
        :param: x, torch.tensor = (n_x, n_features)
        :param: y, torch.tensor = (n_y, n_features)
        :param: square, bool. default is True to follow the setting in 'Prototypical Networks for Few-shot Learning, Snell et al.'
        :return: torch.tensor = (n_x, n_y)
    """
    n_x = x.size(dim=0)
    n_y = y.size(dim=0)

    sub = x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1)

    if square:
        return sub.pow(2).sum(dim=2)
    else:
        return sub.pow(2).sum(dim=2).sqrt()


def cosine_distance(x, y, eps=1e-8):
    """ Calculate cosine distance of `x` and `y`.
        :param: x, torch.tensor = (n_x, n_features)
        :param: y, torch.tensor = (n_y, n_features)
        :param: eps, float, very small number.
    """
    n_x = x.size(dim=0)
    n_y = y.size(dim=0)

    norm_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)
    norm_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)

    cosine_similarity = (norm_x.unsqueeze(1).expand(n_x, n_y, -1) * norm_y.unsqueeze(0).expand(n_x, n_y, -1)).sum(dim=2)

    return 1 - cosine_similarity


def dot(x, y):
    """ Calculate cosine distance of `x` and `y`.
        :param: x, torch.tensor = (n_x, n_features)
        :param: y, torch.tensor = (n_y, n_features)
    """
    n_x = x.size(dim=0)
    n_y = y.size(dim=0)

    return -(x.unsqueeze(1).expand(n_x, n_y, -1) * y.unsqueeze(0).expand(n_x, n_y, -1)).sum(dim=2)


def generate_task_id(num_cls, num_support_per_cls, num_query_per_cls, attach_label=True):
    support_y = create_task_label(num_cls, num_support_per_cls)
    query_y = create_task_label(num_cls, num_query_per_cls)
    y = torch.cat([support_y, query_y], dim=0).to(device)

    return y


def prepare_fewshot_task(num_cls, num_support_per_cls, num_query_per_cls, attach_label=True):
    """ Prepare input data for few shot learning
    :param num_cls: number of class in few shot learning, i.e., 'n ways'
    :param num_support_per_cls: number of support samples per class, i.e., 'K shots'
    :param num_query_per_cls: number of query samples per class
    :param attach_label: bool, if true generate monoline class_idx in an episode corresponding to x
    """
    def _prepare_fewshot_data(x, device):
        """ Create an independent label set for a single few shot learning, e.g., [0, ..., num_cls-1]
            Note that x and y are expected to organise as [support_set, query_set]
        :param x: np.array or torch.tensor
        :param device: CPU/GPU device defined in torch
        :return: (x, y) where y is class_idx in an episode corresponding to x
        """
        if attach_label:
            support_y = create_task_label(num_cls, num_support_per_cls)
            query_y = create_task_label(num_cls, num_query_per_cls)
            y = torch.cat([support_y, query_y], dim=0).to(device)
            return x, y
        else:
            return x

    return _prepare_fewshot_data
# def prepare_fewshot_task(num_cls, num_support_per_cls, num_query_per_cls, attach_label=True):
#     """ Prepare input data for few shot learning
#     :param num_cls: number of class in few shot learning, i.e., 'n ways'
#     :param num_support_per_cls: number of support samples per class, i.e., 'K shots'
#     :param num_query_per_cls: number of query samples per class
#     :param attach_label: bool, if true generate monoline class_idx in an episode corresponding to x
#     """
#     def _prepare_fewshot_data(x, device):
#         """ Create an independent label set for a single few shot learning, e.g., [0, ..., num_cls-1]
#             Note that x and y are expected to organise as [support_set, query_set]
#         :param x: np.array or torch.tensor
#         :param device: CPU/GPU device defined in torch
#         :return: (x, y) where y is class_idx in an episode corresponding to x
#         """
#         if len(x) != (num_cls * (num_support_per_cls + num_query_per_cls)):
#             print("Warning: batch_size is set to {len(x) / (num_cls * (num_support_per_cls + num_query_per_cls)),"
#                   "but only the first batch is processed in the following. }")
#         x = torch.cat(x, dim=0).to(device) # convert to torch.tensor from list and move to proper device
#
#         if attach_label:
#             support_y = create_task_label(num_cls, num_support_per_cls)
#             query_y = create_task_label(num_cls, num_query_per_cls)
#             y = torch.cat([support_y, query_y], dim=0).to(device)
#             return x, y
#         else:
#             return x
#
#     return _prepare_fewshot_data


def create_task_label(num_cls: int, num_sample_per_cls: int) -> Tensor:
    """ Create structured label for an independent task in few shot learning
        Args:
            num_cls: number of class in few shot learning, i.e., 'n ways'
            num_sample_per_cls: number of samples per class, e.g.,, 'K shots' or number of queries
        E.g.:
            [0]*q + [1]*q + ... + [k-1]*q = create_task_label(k, q)
    """
    return torch.arange(0, num_cls, 1 / num_sample_per_cls).long()


def get_prototypes(samples, num_cls, n_samples_per_cls):
    """ Compute prototypes from a cluster of samples.
        :param: samples: torch.tensor = (n_samples_per_cls * num_cls, n_feature).
                expect `samples` are grouped by classes before forwarded, i.e., ([[samples]_cls=0, ..., [samples]]_cls=num_cls-1)
        :param: num_cls: int, expected to be the number of classes in a cluster of samples
        :param: n_samples_per_cls, int, expected to be the number of samples in each class
        :return: torch.tensor = (num_cls, n_feature)
    """
    return samples.reshape(num_cls, n_samples_per_cls, -1).mean(dim=1)


def ML_get_prototypes(samples: Tensor, multihots: Tensor) -> Tensor:
    """ Compute prototypes from a cluster of multi-label samples.
        Args:
            samples (tensor): size = (num_samples, features)
            multihots (tensor): size = (num_samples, num_cls)
        Return:
            tensor: mean of embeddings belonging to this class, size = (num_cls, features)
    """
    _, num_cls = multihots.size()
    _, num_features = samples.size()

    examplars = torch.empty((num_cls, num_features))
    active_indices = (multihots == 1)
    for id in range(num_cls):
        examplars[id] = samples[active_indices[:, id]].mean(dim=0)

    return examplars



def task_label(y: Tensor, *, single_target: bool = True, slt_cls: Optional[Tuple[int, ...]] = None) -> Tensor:
    """ Interface func to order labels in a independent task"""
    if single_target:
        return task_id(y)
    else:
        if slt_cls == None:
            raise ValueError(" `slt_cls` must be given in the mult-target scenario.")
        return task_multihot(y, slt_cls)


def task_id(y: Tensor) -> Tensor:
    """ Convert labels to order labels in a independent task, this function support multi-level label convertion.
        E.p.: [3, 1, 2] = task_label([64, 32, 45])."""
    def _make_task_label(label: Tensor) -> Tensor:
        """ Convert labels to ordered labels in a independent task for few shot learning."""
        _, task_label = torch.unique(label, sorted=False, return_inverse=True)
        return task_label

    if y.dim() == 1:
        y = _make_task_label(y)
    else:
        _, n_level = y.size()
        for lvl in range(n_level):
            y[:, lvl] = _make_task_label(y[:, lvl])
    return y


def task_multihot(y: Tensor, slt_cls: Tuple[int, ...]) -> Tensor:
    """ Convert multihots to ones with limited class number in a independent task
        E.p.: [1, 0, 1, ..., 1] = task_label([1, 0, 1]) with `slt_cls` = [0, 1, 2]."""
    n_sample, _ = y.size()
    n_class = len(slt_cls)
    task_multihot = torch.zeros(n_sample, n_class)
    for i, cls_id in enumerate(slt_cls):
        task_multihot[:, i] = y[:, cls_id]
    return task_multihot


def loss_fn(name: Optional[str] = None, *, single_target: bool = True, reduction: str = 'mean') -> Callable:
    """ Select the loss by passing the name or depending on the type of labels."""
    if name == None:
        if single_target:
            return torch.nn.CrossEntropyLoss(reduction=reduction)
        else:
            return torch.nn.BCEWithLogitsLoss(reduction=reduction)
    elif name == 'CELoss':
        return torch.nn.CrossEntropyLoss(reduction=reduction)
    elif name == 'BCELoss':
        return torch.nn.BCEWithLogitsLoss(reduction=reduction)
    else:
        raise ValueError("Cannot identify the loss type.")


def load_wav(
        wav_path: Union[list, str], *,
        wav_dir: Optional[str] = None,
        sr: int = 44100,
        mono: bool = True
) -> Union[List[Tensor], Tensor]:
    """  Load waveform from the corresponding file path(s)."""
    if isinstance(wav_path, list):
        waveforms = list()
        for wpath in wav_path:
            wpath = os.path.join(wav_dir, wpath) if wav_dir else wpath
            wav, ori_sr = torchaudio.load(wpath, normalize=True)
            wav = wav.squeeze() if (mono == True and ori_sr == sr) else wav
            waveforms.append(wav)
        # Resample the waveform if sample rate are not matched
        if not ori_sr == sr:
            tmp = list()
            resample_fn = torchaudio.transforms.Resample(ori_sr, sr)
            for wav in waveforms:
                w = resample_fn(wav)
                w = w.squeeze() if mono == True else w
                tmp.append(w)
            waveforms = tmp
        return waveforms
    elif isinstance(wav_path, str):
        wav_path = os.path.join(wav_dir, wav_path) if wav_dir else wav_path
        wav, ori_sr = torchaudio.load(wav_path, normalize=True)
        if not ori_sr == sr:
            wav = torchaudio.functional.resample(wav, orig_freq=ori_sr, new_freq=sr)
        wav = wav.squeeze() if mono == True else wav
        return wav
    else:
        raise TypeError(f"Cannot load wav(s) in the type of {type(wav_path)} other than `list` and `str`")


def fix_len(wav: Tensor, win_length: int, overlap: int) -> List[Tensor]:
    """ Make variant-length a waveform into a fixed one by trancating and padding (replicating).
        wav is expected to be a channel_first tensor. """
    clips = list()
    if wav.size(dim=-1) < win_length:
        tmp = _replicate(wav.squeeze(), win_length) # transfer a mono 2-D waveform to 1-D tensor
        clips.append(tmp.unsqueeze(dim=0)) # recover the waveform into size = (n_channel=1, n_samples)
    else:
        hop_length = int(win_length - overlap)
        for idx in range(0, len(wav), hop_length):
            tmp = wav[idx:idx + win_length]
            tmp = _replicate(tmp.squeeze(), win_length)  # to ensure the last seq have the same length
            clips.append(tmp.unsqueeze(dim=0))
    return clips


def _replicate(x: Tensor, min_clip_duration: int) -> Tensor:
    """ Pad a 1-D tensor to fix-length `min_clip_duration` by replicating the existing elements."""
    tile_size = (min_clip_duration // x.size(dim=-1)) + 1
    x = torch.tile(x, dims=(tile_size,))[:min_clip_duration]
    return x


def collate_data(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    """ A customised collate_fn for batched dataloader."""
    _datalist = list()
    _labellist = list()
    for data, label in batch:
        _datalist.append(data)
        _labellist.append(label)
    return torch.stack(_datalist, dim=0), torch.stack(_labellist, dim=0)


class CrossEntropyLabelSmooth(torch.nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


