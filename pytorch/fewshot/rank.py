import torch
from typing import Tuple, Optional, Callable, Union

from pytorch.utils import get_distance, get_prototypes, ML_get_prototypes

torch.multiprocessing.set_sharing_strategy('file_system')


class RankedLoss_onEpisode(object):
    """ Run a ranked loss on episodes."""

    def __init__(
            self, *,
            n_nvl_cls: int,
            n_support_per_cls: int,
            n_query_per_cls: int,
            distance: str = 'l2_distance',
            is_mono: bool = True,
            criterion: Optional[Callable] = None,
            is_train: bool = False,
            optimiser: Optional[torch.optim.Optimizer] = None,
            require_y: bool = False
    ) -> None:
        self.n_nvl_cls = n_nvl_cls
        self.n_support_per_cls = n_support_per_cls
        self.n_query_per_cls = n_query_per_cls
        self.distance = distance
        self.is_mono = is_mono
        self.criterion = criterion
        self.is_train = is_train
        self.optimiser = optimiser
        self.require_y = require_y

    def __call__(
            self,
            model: torch.nn.Module,
            x: torch.Tensor,
            y: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """ This func support is_training on batch by grouping the input in [task_0, ..., task_n], n = len(x) // task_size."""
        device = x.device
        if not device == y.device:
            raise TypeError("x, y must be on the same device.")

        num_samples, _ = x.size()
        task_size = int((self.n_support_per_cls + self.n_query_per_cls) * self.n_nvl_cls)
        if (num_samples % task_size) != 0:
            raise ValueError("cannot create batches automatically, input should be organised in [task_0, ..., task_n]")
        num_task = num_samples // task_size

        if self.is_train:
            model.train()
            self.optimiser.zero_grad()  # zero the param grads
        else:
            model.eval()

        preds, ground_truth, task_loss = list(), list(), list()
        for i in range(num_task):
            embeddings = model(x[i * task_size: (i + 1) * task_size])
            # Embed support and query samples
            supports = embeddings[:self.n_support_per_cls * self.n_nvl_cls]
            queries = embeddings[self.n_support_per_cls * self.n_nvl_cls:]
            # Compute prototypical nodes
            if self.is_mono:
                prototypes = get_prototypes(supports, self.n_nvl_cls, self.n_support_per_cls)
            else:
                prototypes = ML_get_prototypes(supports, y[
                                                         i * task_size: i * task_size + self.n_support_per_cls * self.n_nvl_cls]).to(
                    device)
            # Calculate distance between each query and all prototypes
            distances = get_distance(queries, prototypes, self.distance)
            # Calculate probabilities over queries and loss over one task
            outputs = -distances
            task_loss.append(
                self.criterion(outputs, y[i * task_size + self.n_support_per_cls * self.n_nvl_cls: (i + 1) * task_size])
            )

            if self.require_y:
                o = outputs.detach().softmax(dim=1) if self.is_mono else outputs.detach().sigmoid()
                preds.append(o.cpu())
                ground_truth.append(
                    y[i * task_size + self.n_support_per_cls * self.n_nvl_cls: (i + 1) * task_size].detach().to('cpu'))

        loss = torch.stack(task_loss).mean()
        if self.is_train:
            loss.backward()
            self.optimiser.step()

        if self.require_y:
            return loss.item(), torch.cat(preds, dim=0), torch.cat(ground_truth, dim=0)
        else:
            return loss.item()

    def normalize_rank(x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
          x: pytorch Variable
        Returns:
          x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x

    def euclidean_dist_rank(x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist


def rank_loss(dist_mat, labels, margin, alpha, tval):
    """
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    total_loss = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])

        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]

        ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
        ap_pos_num = ap_is_pos.size(0) + 1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos)
        loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

        an_is_pos = torch.lt(dist_an, alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval * (-1 * an_less_alpha + alpha))
        an_weight_sum = torch.sum(an_weight) + 1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight))
        loss_an = torch.div(an_ln_sum, an_weight_sum)

        total_loss = total_loss + loss_ap + loss_an
    total_loss = total_loss * 1.0 / N
    return total_loss

class RankedLoss(object):
    """ Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper."""
    def __init__(self, margin=None, alpha=None, tval=None):
        self.margin = margin
        self.alpha = alpha
        self.tval = tval

    def __call__(self, global_feat, labels, normalize_feature=True):
        if normalize_feature:
            global_feat = normalize_rank(global_feat, axis=-1)
        dist_mat = euclidean_dist_rank(global_feat, global_feat)
        total_loss = rank_loss(dist_mat, labels, self.margin, self.alpha, self.tval)

        return total_loss
