import torch
from collections import OrderedDict
from torch import Tensor
from typing import Tuple, Optional, Callable, Union

from pytorch.utils import create_task_label

torch.multiprocessing.set_sharing_strategy('file_system')


class MAML_onEpisode(object):
    """ Run a MAML model on episodes.
        Args:
            n_nvl_cls: number of novel classes,
            n_support_per_cls: number of supports per class,
            n_query_per_cls: number of queries per class,
            distance: type of distance function used in metric calculation,
            is_mono: if true, each sample contains only one label,
            is_train: if ture, the model will be updated via batches of input data,
            is_approximate: whether to use 1st-order approximation,
            criterion: loss function used in back propagation,
            optimiser: optimiser used in back propagation,
            base_tr_step: number of training step in a specific task, namely in a base training process,
            base_lr: learning rate in a specific task, namely in a base training process,
            require_y: if true, the function will return ground truth in addition to loss value.
    """
    def __init__(
            self, *,
            n_nvl_cls,
            n_support_per_cls,
            n_query_per_cls,
            distance: str = 'l2_distance',
            is_mono: bool = True,
            is_train: bool = False,
            is_approximate: bool = False,
            criterion: Optional[Callable] = None,
            optimiser: Optional[torch.optim.Optimizer] = None,
            base_tr_step: int = 1,
            base_lr: float = 1e-4,
            require_y: bool = False
    ):
        self.n_nvl_cls = n_nvl_cls
        self.n_support_per_cls = n_support_per_cls
        self.n_query_per_cls = n_query_per_cls
        self.task_size = (self.n_support_per_cls * self.n_query_per_cls) * self.n_nvl_cls
        self.distance = distance
        self.is_mono = is_mono
        self.is_train = is_train
        self.criterion = criterion
        self.optimiser = optimiser
        self.is_approximate = is_approximate
        self.base_tr_step = base_tr_step
        self.base_lr = base_lr
        self.require_y = require_y


    @torch.enable_grad()
    def __call__(
            self,
            model: torch.nn.Module,
            x: Tensor,
            y: Tensor
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        device = x.device
        if not device == y.device:
            raise TypeError("x, y must be on the same device.")

        n_samples, wav_length = x.size()
        task_size = int((self.n_support_per_cls + self.n_query_per_cls) * self.n_nvl_cls)
        if (n_samples % task_size) != 0:
            raise ValueError("cannot create batches automatically, input should be organised in [task_0, ..., task_n]")

        is_create_graph = (True if not self.is_approximate else False) and self.is_train

        if self.require_y:
            ground_truth = list()

        model.train()
        task_grads, task_losses, task_preds = list(), list(), list()
        for i in range(n_samples // task_size):
            # Create a base model with weights of the current meta model
            base_weights = OrderedDict(model.named_parameters())

            # Train the base model with support samples through `self.base_tr_step` iterations
            for _ in range(self.base_tr_step):
                _, _, _gradients = self.step_forward(
                    model,
                    x=x[i*task_size: self.n_support_per_cls*self.n_nvl_cls + i*task_size],
                    y=y[i*task_size: self.n_support_per_cls*self.n_nvl_cls + i*task_size],
                    weights=base_weights,
                    is_retain_graph=True,
                    is_create_graph=is_create_graph
                )

                # Update weights of the base model
                base_weights = OrderedDict(
                    (name, param - self.base_lr * grad) for ((name, param), grad) in zip(base_weights.items(), _gradients)
                )

            # Forward query samples to the updated base model
            outputs, loss, gradients = self.step_forward(
                model,
                x=x[i*task_size + self.n_support_per_cls*self.n_nvl_cls: (i+1)*task_size],
                y=y[i*task_size + self.n_support_per_cls*self.n_nvl_cls: (i+1)*task_size],
                weights=base_weights,
                is_retain_graph=True,
                is_create_graph=is_create_graph
            )
            loss.backward(retain_graph=True)
            # Collect outputs
            task_losses.append(loss)
            _named_grads = {name: grad for (name, grad) in zip(base_weights.keys(), gradients)}
            task_grads.append(_named_grads)

            if self.require_y:
                preds = outputs.detach().cpu().softmax(dim=1)
                task_preds.append(preds)
                ground_truth.append(y[i*task_size + self.n_support_per_cls*self.n_nvl_cls: (i+1)*task_size].detach().to('cpu'))

        meta_loss = torch.stack(task_losses).mean()

        if self.is_train:
            self.optimiser.zero_grad()
            if not self.is_approximate:
                meta_loss.backward()
                self.optimiser.step()

            else:
                avg_task_grads = {k: torch.stack([grad[k] for grad in task_grads]).mean(dim=0)
                                  for k in task_grads[0].keys()}
                hooks = []  # collect hook handlers updating the model using `avg_task_grads`
                for name, param in model.named_parameters():
                    hooks.append(param.register_hook(self.replace_grad(avg_task_grads, name)))

                # Dummy pass to set `loss` variable and replace its gradients with `avg_task_grads`
                _dummy_o = model(torch.zeros((self.n_nvl_cls, wav_length)).to(device, dtype=torch.float))
                loss = self.criterion(_dummy_o, create_task_label(self.n_nvl_cls, 1).to(device))
                loss.backward()
                self.optimiser.step()

                # release hooks
                for h in hooks:
                    h.remove()

        if self.require_y:
            return meta_loss.detach().cpu(), torch.cat(task_preds, dim=0), torch.cat(ground_truth, dim=0)
        else:
            return meta_loss.detach().cpu()


    def step_forward(
            self,
            model: torch.nn.Module,
            x: Tensor,
            y: Tensor,
            weights: OrderedDict,
            is_create_graph: bool,
            is_retain_graph: bool = True
    ):
        """ Calculate loss, grad of a model with a batch of x and y."""
        o = model.functional_forward(x, weights)
        loss = self.criterion(o, y)
        grad = torch.autograd.grad(loss, weights.values(), retain_graph=is_retain_graph, create_graph=is_create_graph)

        return o, loss, grad


    @staticmethod
    def replace_grad(parameter_gradients, parameter_name):
        def _replace_grad(grad):
            return parameter_gradients[parameter_name]

        return _replace_grad