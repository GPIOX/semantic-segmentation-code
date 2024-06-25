
dataset_type = "WaterTankDataset"
root = 'data/sonar'

train_dataset = dict(
    type=dataset_type, 
    root=root,
    split='train'
)
test_dataset = dict(
    type=dataset_type, 
    root=root,
    split='test'
)