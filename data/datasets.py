import os
from PIL import Image

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

    def __init__(self, flist, rootA, rootB, transform=None,
                 flist_loader=default_flist_reader,
                 loader=default_loader):

        self.rootA = rootA
        self.rootB = rootB
        self.loader = loader

        imlist = flist_loader(flist)
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