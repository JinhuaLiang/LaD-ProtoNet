import torch
from typing import Tuple, Optional, List, Callable

from pytorch.utils import get_distance

torch.multiprocessing.set_sharing_strategy('file_system')


class HierProtonet_onEpisode(object):
    """ Run hierarchical prototypical network on episodes.
        This partly refer to 'Leveraging Hierarchical Structures for Few-Shot Musical Instrument Recognition'."""
    def __init__(
            self, *,
            n_nvl_cls: int,
            n_support_per_cls: int,
            n_query_per_cls: int,
            height: int,
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
        self.distance = distance
        self.criterion = criterion
        self.is_train = is_train
        self.optimiser = optimiser
        self.require_y = require_y
        self.height = height
        self.ws = torch.exp(alpha * torch.arange(self.height)) # weight decay to aggregate losses in the hierarchical structure


    def __call__(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,) -> Tuple[torch.Tensor, ...]:
        """ This support training on batch by grouping the input in [task_0, ..., task_n], n = len(x) // task_size."""
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

        embeddings = model(x)

        ground_truth = list()
        preds = list()
        for i in range(num_task):
            # Embed support and query samples
            supports = embeddings[i*task_size :i*task_size+self.n_support_per_cls*self.n_nvl_cls]
            y_supports = y[i*task_size :i*task_size+self.n_support_per_cls*self.n_nvl_cls]
            hier_examplars = self._get_hier_prototypes(supports, y_supports)

            # Calculate loss and predictions in each level
            queries = embeddings[i*task_size+self.n_support_per_cls*self.n_nvl_cls :(i+1)*task_size]
            y_queries = y[i*task_size+self.n_support_per_cls*self.n_nvl_cls :(i+1)*task_size]
            preds = [[] for _ in range(self.height)]
            loss = torch.empty(self.height)
            for level, examplars in enumerate(hier_examplars):
                # Calculate distance between each query and all examplars in the same level
                distances = get_distance(queries, examplars, self.distance)
                # Calculate probabilities over queries and loss over one task
                outputs = -distances
                loss[level] = self.criterion(outputs, y_queries[:, level])
                preds[level].append(outputs.detach().cpu())
            total_loss = torch.dot(loss.T, self.ws)
            # Convert preds to a list where each tensor corresponds to a specific level
            for level, item in enumerate(preds):
                preds[level] = torch.cat(item, dim=0)

            if self.require_y:
                ground_truth.append(y_queries.detach().to('cpu'))

        if self.is_train:
            total_loss.backward()
            self.optimiser.step()

        if self.require_y:
            # only leaf nodes are used for few-shot learning, intermediate nodes are only for auxiliary loss
            return total_loss.item(), preds[0], torch.cat(ground_truth, dim=0)[:, 0]
        else:
            return total_loss.item()


    def _get_hier_prototypes(self, instances: torch.tensor, labels: torch.tensor) -> List[torch.Tensor]:
        """ Calculate hierarchical prototype nodes.
        Args:
            instances (torch.tensor): Embeddings of input data, size = (n_samples, n_features)
            labels (torch.tensor):  (n_samples, n_labels)
        Return:
             hier_prototypes: list of protypical nodes w.r.t. each level,
                len = `self.height`, size of each element = (n_active_classes, n_features)
        """
        device = labels.device
        assert device == instances.device

        _, n_features = instances.size()
        _, n_levels = labels.size()
        if self.height != n_levels:
            print(f"{n_levels} levels obtained from labels but only the first {self.height} are processed.")

        hier_prototypes = list() # [prototype, metaprotorype_0, metaprotorype_1, ...]
        for level in range(self.height):
            labelset = torch.unique(labels[:, level])
            examplars = torch.empty(labelset.size(dim=0), n_features).to(device)
            for i, y in enumerate(labelset):
                examplars[i] = instances[labels[:, level]==y, :].mean(dim=0)
            hier_prototypes.append(examplars)
        return hier_prototypes