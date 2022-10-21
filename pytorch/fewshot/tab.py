import torch
from torch import Tensor
from typing import Tuple, Optional, Callable, Union

from pytorch.utils import get_distance, get_prototypes, ML_get_prototypes

torch.multiprocessing.set_sharing_strategy('file_system')


class TaxonomyBased_onEpisode(object):
    """ Run a prototypical network on episodes.
        It will take as input the output of `HierOne2RestSampler` in the training process,
        and take as input the output of `MultiLabelFewShotSampler` in the evaluation.
    """
    def __init__(
            self, *,
            n_nvl_cls: int,
            n_support_per_cls: int,
            n_query_per_cls: int,
            alpha: int = -1,
            distance: str = 'l2_distance',
            criterion: Optional[Callable] = None,
            is_train: bool = False,
            optimiser: Optional[torch.optim.Optimizer] = None,
            require_y: bool = False
    ) -> None:
        self.n_nvl_cls = n_nvl_cls
        self.n_support_per_cls = n_support_per_cls
        self.n_query_per_cls = n_query_per_cls
        self.alpha = alpha
        self.distance = distance
        self.criterion = criterion
        self.is_train = is_train
        self.optimiser = optimiser
        self.require_y = require_y


    def __call__(
            self,
            model: torch.nn.Module,
            x: Tensor,
            y: Tensor
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """ This func support training on batch by grouping the input in e.g.,
            [task_0, ..., task_n], n = len(x) // task_size."""
        device = x.device
        if not device == y.device:
            raise TypeError("x, y must be on the same device.")

        # Ensure desirable labels are loaded
        onehot_dim = self.n_nvl_cls + 1 if self.is_train else self.n_nvl_cls
        assert y.size(dim=1) == onehot_dim

        num_samples, _ = x.size()
        task_size = int((self.n_support_per_cls + self.n_query_per_cls) * self.n_nvl_cls)
        if (num_samples % task_size) != 0:
            raise ValueError("cannot create batches automatically, input should be organised in [task_0, ..., task_n]")
        num_task = num_samples // task_size

        if self.is_train:
            # Decouple onehot and hierarchy
            onehot = y[:, :-1]
            hierarchy = y[::task_size, -1]
            # Define loss decay as per the level of the label in hierarchy
            decay = torch.exp(self.alpha * (hierarchy - 2))
            model.train()
            self.optimiser.zero_grad()  # zero the param grads
        else:
            onehot = y
            decay = torch.tensor([1] * num_task)
            model.eval()

        for i in range(num_task):
            # Embed support and query samples
            embeddings = model(x[i*task_size: (i+1)*task_size])
            supports = embeddings[:self.n_support_per_cls*self.n_nvl_cls]
            queries = embeddings[self.n_support_per_cls*self.n_nvl_cls:]

            # Compute prototypical nodes
            prototypes = ML_get_prototypes(supports, onehot[i*task_size: i*task_size+self.n_support_per_cls*self.n_nvl_cls]).to(device)

            # Calculate distance between each query and all prototypes
            distances = get_distance(queries, prototypes, self.distance)

            # Calculate probabilities over quries and loss over one task
            outputs = -distances
            loss = self.criterion(outputs, onehot[i*task_size+self.n_support_per_cls*self.n_nvl_cls: (i+1)*task_size]) * decay[i]

            if self.require_y:
                preds, ground_truth = list(), list()
                o = outputs.detach().sigmoid()
                preds.append(o.cpu())
                ground_truth.append(onehot[i*task_size+self.n_support_per_cls*self.n_nvl_cls: (i+1)*task_size].detach().to('cpu'))

        if self.is_train:
            loss.backward()
            self.optimiser.step()

        if self.require_y:
            return loss.item(), torch.cat(preds, dim=0), torch.cat(ground_truth, dim=0)
        else:
            return loss.item()