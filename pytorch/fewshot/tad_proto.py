import torch
from typing import Tuple, Optional, Callable, Union

from pytorch.utils import get_distance, get_prototypes, ML_get_prototypes

torch.multiprocessing.set_sharing_strategy('file_system')


class TaDProtonet_onEpisode(object):
    """ Task-Dependent prototypical network on episodes."""

    def __init__(
            self, *,
            n_nvl_cls: int,
            n_support_per_cls: int,
            n_query_per_cls: int,
            is_aligned: bool = False,
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
        self.is_aligned = is_aligned
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
            if self.is_aligned:
                # Get the sur-label and cut it from `y`
                kp_relation = y[::task_size, -1].long().tolist()
                y = y[:, :-1]
            # Set up model and optimiser
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
                prototypes = ML_get_prototypes(
                    supports,
                    y[i * task_size: i * task_size + self.n_support_per_cls * self.n_nvl_cls]
                ).to(device)
            # Calculate distance between each query and all prototypes
            distances = get_distance(queries, prototypes, self.distance)
            # Calculate probabilities over queries and loss over one task
            outputs = -distances
            # Calculate loss
            _loss = self.criterion(outputs, y[i*task_size+self.n_support_per_cls*self.n_nvl_cls: (i+1)*task_size])
            task_loss.append(_loss)

            if self.require_y:
                o = outputs.detach().softmax(dim=1) if self.is_mono else outputs.detach().sigmoid()
                preds.append(o.cpu())
                ground_truth.append(
                    y[i * task_size + self.n_support_per_cls * self.n_nvl_cls: (i + 1) * task_size].detach().to('cpu')
                )

        if self.is_train:
            if self.is_aligned:
                # Take the max of kid loss and its parental loss
                for kid_id, parent_id in enumerate(kp_relation):
                    if parent_id != -1:
                        task_loss[kid_id] = torch.max(task_loss[kid_id], task_loss[parent_id])
            #     # Del the original parent index
            #     _parent_list = set(kp_relation)
            #     _parent_list.remove(-1)
            #     loss = [_loss for _idx, _loss in enumerate(task_loss) if _idx not in _parent_list]
            #
            # loss = torch.stack(loss).mean()
            loss = torch.stack(task_loss).mean()

            loss.backward()
            self.optimiser.step()

        else:
            loss = torch.stack(task_loss).mean()

        if self.require_y:
            return loss.item(), torch.cat(preds, dim=0), torch.cat(ground_truth, dim=0)
        else:
            return loss.item()


# class TaDProtonet_onEpisode(object):
#     """ Task-Dependent prototypical network on episodes."""
#
#     def __init__(
#             self, *,
#             n_nvl_cls: int,
#             n_support_per_cls: int,
#             n_query_per_cls: int,
#             distance: str = 'l2_distance',
#             is_mono: bool = True,
#             criterion: Optional[Callable] = None,
#             is_train: bool = False,
#             optimiser: Optional[torch.optim.Optimizer] = None,
#             require_y: bool = False
#     ) -> None:
#         self.n_nvl_cls = n_nvl_cls
#         self.n_support_per_cls = n_support_per_cls
#         self.n_query_per_cls = n_query_per_cls
#         self.distance = distance
#         self.is_mono = is_mono
#         self.criterion = criterion
#         self.is_train = is_train
#         self.optimiser = optimiser
#         self.require_y = require_y
#
#     def __call__(
#             self,
#             model: torch.nn.Module,
#             x: torch.Tensor,
#             y: torch.Tensor
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
#         """ This func support is_training on batch by grouping the input in [task_0, ..., task_n], n = len(x) // task_size."""
#         device = x.device
#         if not device == y.device:
#             raise TypeError("x, y must be on the same device.")
#
#         num_samples, _ = x.size()
#         task_size = int((self.n_support_per_cls + self.n_query_per_cls) * self.n_nvl_cls)
#         if (num_samples % task_size) != 0:
#             raise ValueError("cannot create batches automatically, input should be organised in [task_0, ..., task_n]")
#         num_task = num_samples // task_size
#
#         if self.is_train:
#             # Get the sur-label and cut it from `y`
#             pk_relation = y[::task_size, -1].long().tolist()
#             y = y[:, :-1]
#             # Set up model and optimiser
#             model.train()
#             self.optimiser.zero_grad()  # zero the param grads
#         else:
#             model.eval()
#
#         preds, ground_truth, task_loss = list(), list(), list()
#         for i in range(num_task):
#             embeddings = model(x[i * task_size: (i + 1) * task_size])
#             # Embed support and query samples
#             supports = embeddings[:self.n_support_per_cls * self.n_nvl_cls]
#             queries = embeddings[self.n_support_per_cls * self.n_nvl_cls:]
#             # Compute prototypical nodes
#             if self.is_mono:
#                 prototypes = get_prototypes(supports, self.n_nvl_cls, self.n_support_per_cls)
#             else:
#                 prototypes = ML_get_prototypes(supports, y[
#                                                          i * task_size: i * task_size + self.n_support_per_cls * self.n_nvl_cls]).to(
#                     device)
#             # Calculate distance between each query and all prototypes
#             distances = get_distance(queries, prototypes, self.distance)
#             # Calculate probabilities over queries and loss over one task
#             outputs = -distances
#             # Calculate loss
#             _loss = self.criterion(outputs, y[i*task_size+self.n_support_per_cls*self.n_nvl_cls: (i+1)*task_size])
#             if self.is_train and pk_relation[i] != -1:
#                 task_loss.append(torch.tensor(0, device=device))
#                 task_loss[pk_relation[i]] = torch.max(task_loss[pk_relation[i]], _loss)
#             else:
#                 task_loss.append(_loss)
#
#             if self.require_y:
#                 o = outputs.detach().softmax(dim=1) if self.is_mono else outputs.detach().sigmoid()
#                 preds.append(o.cpu())
#                 ground_truth.append(
#                     y[i * task_size + self.n_support_per_cls * self.n_nvl_cls: (i + 1) * task_size].detach().to('cpu'))
#
#         loss = torch.stack(task_loss).mean()
#         if self.is_train:
#             loss.backward()
#             self.optimiser.step()
#
#         if self.require_y:
#             return loss.item(), torch.cat(preds, dim=0), torch.cat(ground_truth, dim=0)
#         else:
#             return loss.item()
