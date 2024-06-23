from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.utils.data as Data
import pytorch_lightning as pl
from mmengine.config import Config, ConfigDict
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)

from base.registry import (MY_DATASETS, MY_TRANSFORM, MY_RUNNER)


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
        self.model = cfg.model

        # Build _build_optimizer
        self.optimizer = self._build_optimizer(self.model, cfg.optim_wrapper)
        self.loss = cfg.loss
        self.optimizer = cfg.optimizer
        self.lr_scheduler = cfg.lr_scheduler

    @staticmethod
    def _build_dataloador(self, 
                          dataset,
                          dataloador_cfg: Optional[Dict] = None):
        return Data.DataLoader(
            dataset,
            **dataloador_cfg
        )
    
    @staticmethod
    def _build_optimizer(self, 
                         model: nn.Module,
                         optim_wrapper: Optional[Union[Optimizer, OptimWrapper, Dict]] = None):
        return build_optim_wrapper(model, optim_wrapper)

    
