import mmengine
from mmengine import Registry

# manage data-related modules
DATASETS = Registry('dataset')
MY_DATASETS = Registry(
    'dataset', parent=DATASETS, locations=['dataset'])
TRANSFORM = Registry('transform')
MY_TRANSFORM = Registry(
    'transform', parent=TRANSFORM, locations=['dataset'])

# manage model-related modules
MODEL = mmengine.MODELS # Registry('runner')
LOSSES = Registry(
    'runner', parent=MODEL, locations=['tools'])

RUNNER = Registry('runner')
MY_RUNNER = Registry(
    'runner', parent=RUNNER, locations=['tools'])
# mangage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = mmengine.PARAM_SCHEDULERS

LOGGER = Registry('logger', locations=['tools'])
