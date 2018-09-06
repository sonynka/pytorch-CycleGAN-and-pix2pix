import os
from PIL import Image
import random
import pandas as pd
import torch

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class AlignedDataset():

    def __init__(self, flist_path, rootA, rootB, transform=None,
                 flist_loader=default_flist_reader,
                 loader=default_loader):

        self.rootA = rootA
        self.rootB = rootB
        self.loader = loader

        imlist = flist_loader(flist_path)
        self.imlist = [im for im in imlist
                       if os.path.exists(os.path.join(rootA, im))
                       & os.path.exists(os.path.join(rootB, im))]

        self.transform = transform

    def __getitem__(self, index):
        pathA = os.path.join(self.rootA, self.imlist[index])
        pathB = os.path.join(self.rootB, self.imlist[index])

        imgA = self.loader(pathA)
        imgB = self.loader(pathB)

        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        return {'A': imgA, 'B': imgB,
                'A_paths': pathA, 'B_paths': pathB}

    def __len__(self):
        return len(self.imlist)

    def name(self):
        return 'AlignedDataset'


class UnalignedDataset():

    def __init__(self, root, flist_path, labels_path, attrA, attrB,
                 categories=None, transform=None, flist_loader=default_flist_reader,
                 loader=default_loader):

        self.root = root
        self.loader = loader
        self.labels_path = labels_path
        self.transform = transform

        imlist = flist_loader(flist_path)
        if categories is not None:
            catlist = categories.split(',')
            imlist = [im for im in imlist if any(cat in im for cat in catlist)]

        self.imlistA, self.imlistB = self.process_labels(attrA, attrB, imlist)

        self.A_size = len(self.imlistA)
        self.B_size = len(self.imlistB)

    def process_labels(self, attrA, attrB, imlist):
        """ Load the labels from CSV file, process, and return a dataframe """

        labels_df = pd.read_csv(self.labels_path)
        labels_df = labels_df.loc[labels_df.img_path.isin(imlist)]

        imlistA = labels_df.loc[labels_df[attrA] == 1].img_path.tolist()
        if attrB is None:
            imlistB = labels_df.loc[labels_df[attrA] == 0].img_path.tolist()
        else:
            imlistB = labels_df.loc[labels_df[attrB] == 1].img_path.tolist()

        return imlistA, imlistB

    def __getitem__(self, index):

        pathA = os.path.join(self.root, self.imlistA[index])
        pathB_idx = random.randint(0, self.B_size - 1)
        pathB = os.path.join(self.root, self.imlistB[pathB_idx])

        imgA = self.loader(pathA)
        imgB = self.loader(pathB)

        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        return {'A': imgA, 'B': imgB,
                'A_paths': pathA, 'B_paths': pathB}

    def __len__(self):
        return min(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'


class LabeledDataset():

    def __init__(self, root, flist_path, labels_path, attributes,
                 categories=None, transform=None, flist_loader=default_flist_reader,
                 loader=default_loader):


        self.root = root
        self.loader = loader
        self.labels_path = labels_path
        self.transform = transform

        imlist = flist_loader(flist_path)
        if categories is not None:
            catlist = categories.split(',')
            imlist = [im for im in imlist if any(cat in im for cat in catlist)]

        self.imgs, self.labels = self.process_labels(imlist, attributes)
        self.size = len(self.imgs)
        self.c_dim = len(self.labels[0])

    def process_labels(self, imlist, attributes):
        """ Load the labels from CSV file, process, and return a dataframe """

        labels_df = pd.read_csv(self.labels_path)
        labels_df = labels_df.loc[labels_df.img_path.isin(imlist)]

        self.selected_attrs = [col for col in labels_df.columns
                               if any(attr in col for attr in attributes.split(','))]

        labels_df = labels_df[['img_path'] + self.selected_attrs]
        labels_df = labels_df.loc[(labels_df[self.selected_attrs] != 0).any(axis=1)]

        imgs = labels_df['img_path'].tolist()
        labels = labels_df.iloc[:, 1:].as_matrix().tolist()

        return imgs, labels


    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.imgs[index])
        img = self.loader(img_path)
        label = torch.FloatTensor(self.labels[index])

        if self.transform is not None:
            img = self.transform(img)

        return {'img_path': img_path, 'img': img, 'label': label}

    def __len__(self):
        return self.size

    def name(self):
        return 'LabeledDataset'