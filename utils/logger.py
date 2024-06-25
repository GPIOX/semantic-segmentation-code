from base.registry import LOGGER
from lightning.pytorch import loggers as pl_loggers

@LOGGER.register_module()
def TensorBoardLogger(**kwargs):
    return pl_loggers.TensorBoardLogger(**kwargs)

@LOGGER.register_module()
def WandbLogger(**kwargs):
    return pl_loggers.WandbLogger(**kwargs)

@LOGGER.register_module()
def CometLogger(**kwargs):
    return pl_loggers.CometLogger(**kwargs)

@LOGGER.register_module()
def CSVLogger(**kwargs):
    return pl_loggers.CSVLogger(**kwargs)

