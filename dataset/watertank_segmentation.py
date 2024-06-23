
from dataset.base import BaseDataset
from base.registry import MY_DATASETS

@MY_DATASETS.register_module()
class WaterTankDataset(BaseDataset):
    METAINFO = dict(
        classes=('Background', 'Bottle', 'Can', 'Chain',
                 'Drink-carton', 'Hook', 'Propeller', 'Shampoo-bottle' ,
                 'Standing-bottle', 'Tire', 'Valve', 'Wall'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])
    def __init__(self, 
                 root, 
                 transform=None, 
                 target_transform=None, 
                 **kwargs):
        super().__init__(
            root=root, 
            transform=transform, 
            target_transform=target_transform, 
            **kwargs)
