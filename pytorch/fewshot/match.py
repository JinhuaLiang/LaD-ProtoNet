import torch
from torch.nn.functional import one_hot
from typing import Tuple, Optional, Callable, Union

from pytorch.utils import get_distance

torch.multiprocessing.set_sharing_strategy('file_system')


class Matchnet_onEpisode(object):
    """ Run a Matching network on episodes."""

    def __init__(
            self, *,
            n_nvl_cls: int,
            n_support_per_cls: int,
            n_query_per_cls: int,
            distance: str = 'cosine',
            is_mono: bool = True,
            criterion: Optional[Callable] = None,
            is_train: bool = False,
            optimiser: Optional[torch.optim.Optimizer] = None,
            require_y: bool = False
    ):
        self.n_nvl_cls = n_nvl_cls
        self.n_support_per_cls = n_support_per_cls
        self.n_query_per_cls = n_query_per_cls
        self.distance = distance
        self.is_mono = is_mono
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            self.criterion = torch.nn.NLLLoss(reduction=criterion.reduction)
        elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            self.criterion = torch.nn.BCELoss(reduction=criterion.reduction)
        else:
            raise ValueError("Cannot parse the passed loss function.")

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

        embeddings = model.encoder(x)

        preds, ground_truth, task_loss = list(), list(), list()
        for i in range(num_task):
            # Embed supports and query samples
            supports = embeddings[i*task_size: i*task_size+self.n_support_per_cls*self.n_nvl_cls]
            queries = embeddings[i*task_size+self.n_support_per_cls*self.n_nvl_cls: (i+1)*task_size]

            y_supports = y[i*task_size: i*task_size+self.n_support_per_cls*self.n_nvl_cls]
            y_queries = y[i*task_size+self.n_support_per_cls*self.n_nvl_cls: (i+1)*task_size]
            # Fully condition embedding
            # Convert `support` to the shape of (self.n_support_per_cls*self.n_nvl_cls, batch_size(=1), embedding_dim)
            # to suit LSTM func.
            # Calculate the fully conditional embedding g, for support set samples
            # as described in appendix A.2 of the paper.
            supports, _, _ = model.g(supports.unsqueeze(1))
            # Remove the batch dimension after feeding to LSTM
            supports = supports.squeeze(1)
            # Calculate the fully conditional embedding, f, for the query set samples as described in appendix A.1 of the paper.
            queries = model.f(supports, queries)
            # Calculate distance between each query and all support samples
            distances = get_distance(queries, supports, self.distance)
            if self.is_mono:
                # Calculate "attention" as softmax over query-support distances
                attention = (-distances).softmax(dim=1)
                # Calculate predictions as in equation (1) from Matching Networks: y_hat = \sum_{i=1}^{k} a(x_hat, x_i) y_i
                # `y` should be organised in one hot format
                outputs = torch.mm(attention, one_hot(y_supports, self.n_nvl_cls).float().to(device))

            else:
                attention = (-distances).sigmoid()
                outputs = torch.mm(attention, y_supports.float().to(device))

            outputs = outputs.clamp(1e-8, 1 - 1e-8)  # clip predictions for numerical stability, EPSILON = 1e-8
            task_loss.append(self.criterion(outputs.log(), y_queries))

            if self.require_y:
                preds.append(outputs.detach().cpu())
                ground_truth.append(
                    y[i*task_size+self.n_support_per_cls*self.n_nvl_cls: (i+1)*task_size].detach().to('cpu')
                )

        loss = torch.stack(task_loss).mean()
        if self.is_train:
            loss.backward()
            self.optimiser.step()

        if self.require_y:
            return loss.item(), torch.cat(preds, dim=0), torch.cat(ground_truth, dim=0)
        else:
            return loss.item()
