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
        self.optim_wrapper = self._build_optimizer(self.model, cfg.optim_wrapper)
        self.param_schedulers = self._build_param_scheduler(self.optim_wrapper, cfg.param_scheduler)

        # Save predict outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, inputs):
        img = inputs['img']
        mask = inputs['mask']
        output = self.model(img)
        
        loss, log_vars = self._parse_losses(self.loss_decode, output, mask)

        return loss, log_vars, output
    def training_step(self, batch, batch_idx):
        """
        训练步骤，执行一次训练迭代。
        
        在这个步骤中，首先通过调用self函数进行前向传播，计算损失、日志变量和输出。
        然后，根据计算得到的损失更新网络参数。
        接着，记录当前的训练损失和学习率。
        最后，对每个学习率调度器调用step方法。
        
        参数:
            - batch: 当前批次的数据。
            - batch_idx: 当前批次的索引。
        
        返回:
            - log_vars: 包含日志变量的字典。
        """
        loss, log_vars, output = self(batch)
        self.optim_wrapper.update_params(loss)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        lr = self.optim_wrapper.optimizer.param_groups[0]['lr']
        self.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        for _scheduler in self.param_schedulers:
            _scheduler.step()

        return log_vars
    
    def validation_step(self, batch, batch_idx):
        loss, log_vars, output = self(batch)
        
        self.log("val_loss", loss, logger=True, on_step=True)

    def on_validation_epoch_end(self):
        # all_preds = torch.stack(self.validation_step_outputs)
        # # do something with all preds
        # self.validation_step_outputs.clear()  # free memory
        pass

    def test_step(self, batch, batch_idx):
        loss, log_vars, output = self(batch)
        self.log("test_loss", loss, logger=True, on_step=True)
        # pass

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
        """
        构建参数调度器列表。

        该方法用于根据传入的参数调度器配置，创建并返回一个参数调度器列表。
        参数调度器用于在训练过程中动态调整模型参数的学习率或其他属性。

        参数:
            - optimizer: 优化器实例，参数调度器将与之关联。
            - param_scheduler: 可选的参数调度器配置，可以是一个参数调度器实例或一个字典。
        
        返回:
            - List[_ParamScheduler]: 一个包含构建的参数调度器实例的列表。
        """
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
