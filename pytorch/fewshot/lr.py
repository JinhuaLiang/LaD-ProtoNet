import torch
from collections import OrderedDict
from torch import Tensor
from typing import Tuple, Optional, Callable, Union

torch.multiprocessing.set_sharing_strategy('file_system')


class LinearRegression(object):
    """ Run a linear regression.
        Note that lr methods are based upon a pretrained model, so we assume user won't use it in a meta learning way.
        Args:
            n_nvl_cls: number of novel classes,
            n_support_per_cls: number of supports per class,
            n_query_per_cls: number of queries per class,
            is_mono: if true, each sample contains only one label,
            criterion: loss function used in back propagation,
            optimiser: optimiser used in back propagation,
            freeze_encoder: If true, finetune the logit layer only,
            base_tr_step: number of training step in a specific task, namely in a base training process,
            require_y: if true, the function will return ground truth in addition to loss value.
    """
    def __init__(
            self, *,
            n_nvl_cls,
            n_support_per_cls,
            n_query_per_cls,
            is_mono: bool = True,
            criterion: Optional[Callable] = None,
            optimiser: Optional[torch.optim.Optimizer] = None,
            freeze_encoder: bool = False,
            base_tr_step: int = 1,
            require_y: bool = False
    ):
        self.n_nvl_cls = n_nvl_cls
        self.n_support_per_cls = n_support_per_cls
        self.n_query_per_cls = n_query_per_cls
        self.task_size = (self.n_support_per_cls * self.n_query_per_cls) * self.n_nvl_cls
        self.is_mono = is_mono
        self.criterion = criterion
        self.optimiser = optimiser
        self.freeze_encoder = freeze_encoder
        self.base_tr_step = base_tr_step
        self.require_y = require_y


    def __call__(
            self,
            model: torch.nn.Module,
            x: Tensor,
            y: Tensor
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        device = x.device
        if not device == y.device:
            raise TypeError("x, y must be on the same device.")

        # check if `n_samples` == `task_size`
        if x.size()[1] != int((self.n_support_per_cls + self.n_query_per_cls) * self.n_nvl_cls):
            raise ValueError("Linear regression only take one FSL task a time.")

        if self.require_y:
            ground_truth = list()

        # save initial weights
        torch.save({'model': model.state_dict(),
                    'optimiser': self.optimiser.state_dict()
                    }, 'tmpCkpt.pth')

        if self.freeze_encoder:
            for name, param in model.named_parameters():
                if name not in ['logits.weight', 'logits.bias']:
                    param.requires_grad = False

        # Fine-tune a pretrained model `base_tr_step` times
        model.train()
        for _ in range(self.base_tr_step):
            _output = model(x[: self.n_support_per_cls*self.n_nvl_cls])
            _loss = self.criterion(_output, y[: self.n_support_per_cls*self.n_nvl_cls])
            _loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

        # Forward query samples to the fine-tuned model
        model.eval()
        output = model(x[self.n_support_per_cls*self.n_nvl_cls:])
        loss = self.criterion(output, y[self.n_support_per_cls*self.n_nvl_cls:])

        # Restore params of model and optimiser
        ckpt = torch.load('tmpCkpt.pth')
        model.load_state_dict(ckpt['model'])
        self.optimiser.load_state_dict(ckpt['optimiser'])
        os.remove('tmpCkpt.pth')  # release the space of the tmp file

        if self.require_y:
            preds = output.detach().cpu().softmax(dim=1) if self.is_mono else output.detach().cpu().sigmoid()
            return loss.item(), preds, y[self.n_support_per_cls*self.n_nvl_cls:].detach().to('cpu')

        else:
            return loss.item()
