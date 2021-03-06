import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class GatedGANModel(BaseModel):
    """
    This class implements the GatedGANModel model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    GatedGAN paper: https://arxiv.org/pdf/1904.02296.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G', 'g', 'AC', 'rec', 'tv']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        if self.isTrain:
            self.visual_names.append('rec')
        else:
            self.visual_names += ['fake_B%d' % i for i in range(opt.n_style)]

        # define networks (both Generators and discriminators)
        self.netG_A = networks.define_Gated_G(opt.input_nc, opt.n_style, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.n_style)

        if self.isTrain:
            self.autoflag = torch.zeros(opt.batch_size, opt.n_style + 1)
            self.autoflag[:, -1] = 1
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionRec = torch.nn.MSELoss() if opt.l2_loss else torch.nn.L1Loss() # auto encoder reconstruction loss
            self.criterionAC = torch.nn.CrossEntropyLoss() # auxiliary classifier loss
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.class_label_B = input['B_style_labels'].to(self.device)
        self.one_hot_label = torch.zeros(self.opt.batch_size, self.opt.n_style + 1).to(self.device)
        self.one_hot_label.scatter_(1, self.class_label_B.unsqueeze(1), 1)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A, self.one_hot_label)  # G_A(A)
        if self.isTrain:
            self.rec = self.netG_A(self.real_A, self.autoflag, True) # autoencoder
            assert self.rec.shape == self.real_A.shape
        else:
            for i in range(self.opt.n_style):
                flag = torch.zeros(self.opt.batch_size, self.opt.n_style + 1)
                flag[:, i] = 1
                setattr(self, 'fake_B%d' % i, self.netG_A(self.real_A, flag))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        prediction, classification = self.netD_A(real)
        loss_d_real = self.criterionGAN(prediction, True)
        N, _, H, W = classification.shape
        expanded_label = self.class_label_B.unsqueeze(1).unsqueeze(2).expand(N, H, W)
        loss_AC_real = self.criterionAC(classification, expanded_label)

        prediction, classification = self.netD_A(fake)
        loss_d_fake = self.criterionGAN(prediction, False)
        loss_AC_fake = self.criterionAC(classification, expanded_label) # TODO: this was commented out in original implementation
        loss_D = loss_d_real + loss_AC_real + loss_d_fake + loss_AC_fake
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_G(self):
        """Calculate the loss for generators G_A"""
        autoencoder_constraint = self.opt.autoencoder_constraint
        lambda_A = self.opt.lambda_A

        # auto-encoder loss
        self.loss_rec = autoencoder_constraint * self.criterionRec(self.rec, self.real_A)

        # gan loss
        prediction, classification = self.netD_A(self.fake_B)
        self.loss_g = self.criterionGAN(prediction, True)
        N, _, H, W = classification.shape
        expanded_label = self.class_label_B.unsqueeze(1).unsqueeze(2).expand(N, H, W)
        self.loss_AC = self.criterionAC(classification, expanded_label)
        
        # total variation loss
        if self.opt.tv_strength > 0:
            self.loss_tv = torch.sqrt(torch.sum((self.fake_B[:, :, :, :-1] - self.fake_B[:, :, :, 1:]) ** 2) 
            + torch.sum((self.fake_B[:, :, :-1, :] - self.fake_B[:, :, 1:, :]) ** 2))
        else:
            self.loss_tv = 0
        self.loss_G = self.loss_g + self.loss_AC * lambda_A + self.loss_rec + self.loss_tv * self.opt.tv_strength
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
