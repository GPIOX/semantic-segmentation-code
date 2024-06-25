from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Union, Tuple
import copy

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.utils.data as Data
import pytorch_lightning as pl
from mmengine.config import Config, ConfigDict
from mmengine.utils import is_list_of
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)

from base.registry import (MY_DATASETS, MY_TRANSFORM, MY_RUNNER, PARAM_SCHEDULERS, MODEL)
from base.registry import LOSSES
import model as Modelzoo
import utils.loos
ConfigType = Union[Dict, Config, ConfigDict]

@MY_RUNNER.register_module()
class Runner(pl.LightningModule):
    cfg: Config
    def __init__(
        self, 
        work_dir: str = None, 
        cfg: Optional[ConfigType] = None, 
        **kwargs
    ):
        super().__init__()
        # Build dataloader
        self.cfg = cfg
        self.workdir = work_dir
        train_dataset = MY_DATASETS.build(cfg.train_dataset)
        test_dataset = MY_DATASETS.build(cfg.test_dataset)
        self._train_dataloader = self._build_dataloador(train_dataset, 
                                                        cfg.train_dataloader)
        self._test_dataloader = self._build_dataloador(test_dataset, 
                                                       cfg.test_dataloader)

        # Build model
        self.model = self._build_model_from_cfg(cfg.model_cfg) # cfg.model

        # Build loss
        self.loss_decode = self._build_loss_from_cfg(cfg.decode_loss)

        # Build _build_optimizer
        self.optimizer = self._build_optimizer(self.model, cfg.optim_wrapper)
        self.param_schedulers = self._build_param_scheduler(self.optimizer, cfg.param_scheduler)

        self.validation_step_outputs = []

    def forward(self, inputs):
        img = inputs['img']

        return self.model(inputs)
    def training_step(self, batch, batch_idx):
        img = batch_idx['img']
        mask = batch_idx['mask']

        output = self(img)

        loss, log_vars = self._parse_losses(self.loss_decode, output, mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return log_vars
    
    def validation_step(self, batch, batch_idx):
        img = batch_idx['img']
        mask = batch_idx['mask']

        output = self(img)

        loss, log_vars = self._parse_losses(self.loss_decode, output, mask)
        self.log("val_loss", loss)

    def on_validation_epoch_end(self):
        # all_preds = torch.stack(self.validation_step_outputs)
        # # do something with all preds
        # self.validation_step_outputs.clear()  # free memory
        pass

    def test_step(self, batch, batch_idx):
        img = batch_idx['img']
        mask = batch_idx['mask']

        output = self(img)

        loss, log_vars = self._parse_losses(self.loss_decode, output, mask)
        self.log("test_loss", loss)

    def on_test_epoch_end(self):
        # all_preds = torch.stack(self.validation_step_outputs)
        # # do something with all preds
        # self.validation_step_outputs.clear()  # free memory
        pass

    def _build_dataloador(self, 
                          dataset: Data.Dataset,
                          dataloador_cfg: Optional[Dict] = None):
        return Data.DataLoader(
            dataset,
            **dataloador_cfg
        )
    
    # @staticmethod
    def _build_optimizer(self, 
                         model: nn.Module,
                         optim_wrapper: Optional[Union[Optimizer, OptimWrapper, Dict]] = None
        ) -> OptimWrapper:
        return build_optim_wrapper(model, optim_wrapper)
    
    # @staticmethod
    def _build_param_scheduler(self, 
                            optimizer: OptimWrapper,
                            param_scheduler: Optional[Union[_ParamScheduler, Dict]] = None
        ) -> List[_ParamScheduler]:
        param_schedulers = []
        for scheduler in param_scheduler:
            if isinstance(scheduler, dict):
                _scheduler = copy.deepcopy(scheduler)
                param_schedulers.append(
                        PARAM_SCHEDULERS.build(
                            _scheduler,
                            default_args=dict(
                                optimizer=optimizer,
                                epoch_length=123)))
        
        return param_schedulers

    def _build_loss_from_cfg(self, 
                             loss_cfg: Dict):
        if isinstance(loss_cfg, dict):
            return LOSSES.build(loss_cfg)
    
    def _build_model_from_cfg(self, 
                            model_cfg: Dict):
        if isinstance(model_cfg, dict):
            return MODEL.build(model_cfg)
        
    def _parse_losses(
        self, loss_decode, 
        seg_logits: torch.Tensor, 
        seg_label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        loss = dict()
        for _decode in loss_decode:
            if _decode.loss_name not in loss:
                loss[_decode.loss_name] = _decode(
                    seg_logits,
                    seg_label)
            else:
                loss[_decode.loss_name] += _decode(
                    seg_logits,
                    seg_label)
        # loss['acc_seg'] = accuracy(
        #     seg_logits, seg_label, ignore_index=self.ignore_index)

        log_vars = []
        for loss_name, loss_value in loss.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars  # type: ignore
