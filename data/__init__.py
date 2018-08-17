from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from data import datasets
import os

def get_data_loaders(opt):

    modes = ['train', 'val', 'test']

    if opt.dataset_mode == 'aligned':
        dataset = datasets.AlignedDataset

    data_transforms = \
        transforms.Compose([
            transforms.Resize(opt.image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image_datasets = {
        mode: dataset(os.path.join(opt.flist_path, mode + '_imgs.csv'),
                      opt.datarootA, opt.datarootB, transform=data_transforms)
        for mode in modes
    }

    data_loaders = {
        mode: DataLoader(image_datasets[mode],
                         batch_size=16 if mode =='val' else opt.batch_size,
                         shuffle=True if mode == 'train' else False,
                         num_workers=4)
        for mode in modes
    }

    return data_loaders
