from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from data import datasets
import os

from logging import getLogger

logger = getLogger()

def get_data_loaders(opt, modes=None):

    if not modes:
        modes = ['train', 'val', 'test']

    d_args = {}
    dataset = None
    if opt.dataset_mode == 'aligned':
        dataset = datasets.AlignedDataset
        d_args = {'rootA': opt.datarootA, 'rootB': opt.datarootB}
    elif opt.dataset_mode == 'unaligned':
        dataset = datasets.UnalignedDataset
        d_args = {'root': opt.dataroot,
                  'labels_path': os.path.join(opt.dataroot, 'img_attr.csv'),
                  'attrA': opt.attrA, 'attrB': opt.attrB,
                  'categories': opt.categories}
    elif opt.dataset_mode == 'labeled':
        dataset = datasets.LabeledDataset
        d_args = {'root': opt.dataroot,
                  'labels_path': os.path.join(opt.dataroot, 'img_attr.csv'),
                  'attributes': opt.attributes,
                  'categories': opt.categories}

    data_transforms = \
        transforms.Compose([
            transforms.Resize(opt.image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    image_datasets = {
        mode: dataset(
            flist_path=os.path.join(opt.flist_path, mode + '_imgs.csv'),
            transform=data_transforms,
            **d_args)
        for mode in modes
    }

    data_loaders = {
        mode: DataLoader(image_datasets[mode],
                         batch_size=8 if mode =='val' else opt.batch_size,
                         shuffle=False if mode == 'test' else True,
                         num_workers=4,
                         drop_last=True)
        for mode in modes
    }

    logger.info('--- Loaded Datasets ---')
    for mode in modes:
        logger.info('{} size: {}'.format(mode, len(image_datasets[mode])))

    return data_loaders
