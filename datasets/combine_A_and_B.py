import os
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)',action='store_true')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

img_files = os.listdir(args.fold_A)
num_imgs = len(img_files)
valid_AB_imgs = []

for n in range(num_imgs):
    name_A = img_files[n]
    path_A = os.path.join(args.fold_A, name_A)
    if args.use_AB:
        name_B = name_A.replace('_A.', '_B.')
    else:
        name_B = name_A
    path_B = os.path.join(args.fold_B, name_B)

    if os.path.isfile(path_A) and os.path.isfile(path_B):
        name_AB = name_A
        if args.use_AB:
            name_AB = name_AB.replace('_A.', '.') # remove _A
        valid_AB_imgs.append(name_AB)

num_valid_imgs = len(valid_AB_imgs)
train_idx = int(num_valid_imgs*0.8)
valid_idx = int(num_valid_imgs*0.9)
splits = {'train': valid_AB_imgs[:train_idx],
          'valid': valid_AB_imgs[train_idx:valid_idx],
          'test': valid_AB_imgs[valid_idx:]}

for split, split_files in splits.items():
    folder_AB = os.path.join(args.fold_AB, split)
    if not os.path.exists(folder_AB):
        os.makedirs(folder_AB)

    for img_name in split_files:
        path_AB = os.path.join(folder_AB, img_name)
        path_A = os.path.join(args.fold_A, img_name)
        path_B = os.path.join(args.fold_B, img_name)
        im_A = Image.open(path_A)
        im_B = Image.open(path_B)
        im_AB = np.concatenate([im_A, im_B], 1)
        Image.fromarray(im_AB).save(path_AB)
