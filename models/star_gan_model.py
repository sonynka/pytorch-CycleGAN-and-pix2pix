import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np


class StarGANModel(BaseModel):
    def name(self):
        return 'StarGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='stargan')
        parser.set_defaults(which_model_netD='stargan')
        parser.set_defaults(n_layers_D=6)
        parser.add_argument('--c_dim', type=int, default=5,
                            help='number of classes')

        if is_train:
            parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
            parser.add_argument('--lambda_cls', type=float, default=1, help='weight for classification loss')
            parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty loss')
            parser.add_argument('--lambda_l1', type=float, default=10, help='weight of l1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D', 'D_real', 'D_fake', 'D_cls', 'D_gp',
                           'G', 'G_GAN', 'G_cls', 'G_rec', 'G_L1']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_img', 'fake_img', 'rec_img']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc + opt.c_dim, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout,
                                      opt.init_type, self.gpu_ids, opt.image_size)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.c_dim, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.init_type, self.gpu_ids, opt.image_size)

        if self.isTrain:
            self.fake_img_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(soft_labels=True).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionBCE = networks.ClassificationLoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.fixed_labels = self.make_fixed_labels()

    def make_fixed_labels(self):
        """Generate domain labels for dataset for debugging/testing.
        """
        fixed_labels = []
        for dim in range(self.opt.c_dim):
            t = [0] * self.opt.c_dim
            t[dim] = 1
            t = torch.FloatTensor(t).expand([self.opt.batch_size, self.opt.c_dim])
            fixed_labels.append(t)
        return fixed_labels

    def set_input(self, input):
        self.real_img = input['img'].to(self.device)
        self.real_label = input['label'].to(self.device)
        self.image_paths = input['img_path']

        rand_idx = torch.randperm(self.real_label.size(0))
        self.fake_label = self.real_label[rand_idx]

    def forward(self, use_fixed_labels=False):
        self.fake_img = self.netG(self.real_img, self.fake_label)
        self.rec_img = self.netG(self.fake_img, self.real_label)

        if use_fixed_labels:
            fake_fixed_img = []
            for fixed_label in self.fixed_labels:
                fake_fixed_img.append(self.netG(self.real_img, fixed_label))
            self.fake_img = torch.cat(fake_fixed_img, dim=2)

    def backward_D(self, flip=False):
        # Sometimes flip labels to confuse the discriminator
        if flip:
            flip_labels = np.random.rand()
            flip_labels = True if flip_labels < 0.05 else False
        else:
            flip_labels = False

        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_img = self.fake_img_pool.query(self.fake_img)
        pred_fake, _ = self.netD(fake_img.detach())

        self.loss_D_fake = self.criterionGAN(pred_fake, flip_labels)

        # Real
        pred_real, pred_label = self.netD(self.real_img)
        self.loss_D_real = self.criterionGAN(pred_real, not flip_labels)
        self.loss_D_cls = self.criterionBCE(
            pred_label, self.real_label, self.opt.batch_size) * self.opt.lambda_cls

        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_D_cls
        self.loss_D.backward()
        self.optimizer_D.step()

        # Wasserstein
        alpha = torch.rand(self.opt.batch_size, 1, 1, 1).to(self.device)\
                     .expand_as(self.real_img)
        interpolated = torch.autograd.Variable(
            alpha * self.real_img.detach() + ((1 - alpha) * self.fake_img.detach()),
            requires_grad=True).to(self.device)

        pred_intp, pred_label = self.netD(interpolated)

        grad = torch.autograd.grad(outputs=pred_intp,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(pred_intp.size()).to(self.device),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)[0]
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        self.loss_D_gp = torch.mean((grad_l2norm - 1) ** 2) * self.opt.lambda_gp

        self.optimizer_D.zero_grad()
        self.loss_D_gp.backward()
        self.optimizer_D.step()

        # Combined loss
        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_D_cls + self.loss_D_gp

    def backward_G(self):
        # First, G(A) should fake the discriminator
        pred_fake, pred_label = self.netD(self.fake_img)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_cls = self.criterionBCE(
            pred_label, self.fake_label, self.opt.batch_size) * self.opt.lambda_cls

        # Second, reconstructed image = real image
        self.loss_G_rec = self.criterionL1(
            self.rec_img, self.real_img) * self.opt.lambda_rec

        self.loss_G_L1 = self.criterionL1(self.fake_img, self.real_img) * self.opt.lambda_l1

        self.loss_G = self.loss_G_GAN + self.loss_G_rec + self.loss_G_cls + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self, optimize_G=True):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()

        # update G
        if optimize_G:
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
