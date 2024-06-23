from mmengine import Registry

# manage data-related modules
DATASETS = Registry('dataset')
MY_DATASETS = Registry(
    'dataset', parent=DATASETS, locations=['dataset'])
TRANSFORM = Registry('transform')
MY_TRANSFORM = Registry(
    'transform', parent=TRANSFORM, locations=['dataset'])

# manage model-related modules
MODEL = Registry('runner')
MY_MODEL = Registry(
    'runner', parent=MODEL, locations=['tools'])

RUNNER = Registry('runner')
MY_RUNNER = Registry(
    'runner', parent=RUNNER, locations=['tools'])