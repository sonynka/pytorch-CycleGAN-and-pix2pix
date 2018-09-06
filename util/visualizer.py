import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from scipy.misc import imresize
from util.tensorboard_logger import Logger
from util.logger import create_logger
import torch
from datetime import datetime


# save image to the disk
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    def __init__(self, opt):
        self.opt = opt

        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        self.log_dir = os.path.join(opt.log_dir, opt.name)

        util.mkdirs([self.web_dir, self.img_dir])

        log_name = 'train{}.log'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.logger = create_logger(os.path.join(self.log_dir, log_name))
        self.logger.info('============ Initialized logger ============')
        self.logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                                    in sorted(dict(vars(opt)).items())))

        self.tb_logger = Logger(os.path.join(opt.log_dir, opt.name))


    # |visuals|: dictionary of images to display or save
    def log_current_visuals(self, visuals, epoch, step):
        imgs = []

        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            util.save_image(image_numpy, img_path)

            imgs.append(image_numpy)

        self.tb_logger.image_summary(visuals.keys(), imgs, step)

    # losses: same format as |losses| of plot_current_losses
    def log_current_losses(self, epoch, i, losses, step):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for tag, value in losses.items():
            message += '%s: %.3f ' % (tag, value)
            self.tb_logger.scalar_summary(tag, value, step)

        self.logger.info(message)
