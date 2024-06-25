import torch
import cv2
import torch.utils.data
import numpy as np
import os
import os.path as osp
from base.registry import MY_DATASETS
from torchvision import transforms

_CLASS_NAME_ = [
    'Background',
    'Bottle',
    'Can',
    'Chain',
    'Drink-carton',
    'Hook',
    'Propeller',
    'Shampoo-bottle' ,
    'Standing-bottle',
    'Tire',
    'Valve',
    'Wall'
]
def load_txt(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

# 注册模块，使得该类可以被MY_DATASETS框架识别和使用
@MY_DATASETS.register_module()
class BaseDataset(torch.utils.data.Dataset):
    """
    基础数据集类，用于加载和处理图像和掩码数据。
    
    参数:
    - root: 数据集根目录的路径。
    - transform: 图像转换操作，可选。
    - target_transform: 目标（掩码）转换操作，可选。
    - split: 数据集划分，默认为'train'，可选。
    - **kwargs: 其他关键字参数，用于扩展。
    """
    def __init__(self, root, transform=None, target_transform=None, split='train', **kwargs):
        """
        初始化数据集类，设置数据路径和文件列表。
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        # 图像路径
        self.imgs_path = os.path.join(root, "Images")
        # 掩码路径
        self.masks = os.path.join(root, "Masks")
        # 根据split加载对应的文件列表
        # 读取root下的train.txt文件
        self.file_list = load_txt(os.path.join(root, f"{split}.txt"))

        if self.transform is None or self.target_transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.target_transform = torch.from_numpy

    def __getitem__(self, index):
        """
        根据索引获取数据集中的一个样本，包括图像和对应的掩码。
        
        返回:
        - output: 包含图像和掩码的字典。
        """
        output = {} # define return dict
        img_path = os.path.join(self.imgs_path, self.file_list[index])
        mask_path = os.path.join(self.masks, self.file_list[index])
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        img = self.transform(img)
        mask = self.target_transform(mask).long()
        # if len(mask.shape) == 2:
        #     mask = mask.unsqueeze(0)
        
        # 将处理后的图像和掩码添加到输出字典中
        output['img'] = img # []
        output['mask'] = mask
        output['filename'] = self.file_list[index]

        return output

    def __len__(self):
        """
        返回数据集的大小，即文件列表的长度。
        """
        return len(self.file_list)

if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = BaseDataset(transform=transform, root='data/watertank-segmentation', split='train')
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for i, output in enumerate(dataloader):
        assert output['img'].shape == output['mask'].shape, print(output['filename'])
        