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
def save_images(img_dir, visuals, image_path):

    img_name = os.path.basename(image_path)

    for label, image in visuals.items():
        image_numpy = util.tensor2im(image)
        img_path = os.path.join(img_dir, '{}_{}.png'.format(img_name, label))
        util.save_image(image_numpy, img_path)


class Visualizer():
    def __init__(self, opt):
        self.opt = opt

        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        self.log_dir = os.path.join(opt.log_dir, opt.name)

        util.mkdirs([self.web_dir, self.img_dir, self.log_dir])

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
