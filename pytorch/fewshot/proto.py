import torch
from typing import Tuple, Optional, Callable, Union

from pytorch.utils import get_distance, get_prototypes, ML_get_prototypes

torch.multiprocessing.set_sharing_strategy('file_system')


class Protonet_onEpisode(object):
    """ Run a prototypical network on episodes."""

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
