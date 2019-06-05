import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from scipy import ndimage as ndi
import numpy as np

# TODO: move these to seperate files
# Code based from https://github.com/heitorrapela/xdog/blob/master/main.py with modifications
def dog(blur1, blur2, imgs, gamma=1):
    return blur1(imgs) - gamma * blur2(imgs)

# Code based from https://github.com/heitorrapela/xdog/blob/master/main.py with modifications
def xdog(blur1, blur2, imgs, gamma=1, epsilon=-1, phi=1):
    aux = dog(blur1, blur2, imgs, gamma)
    mask = aux < epsilon
    aux = torch.tanh(phi * (aux - epsilon)) + 1
    aux[mask] = 1
    return aux

# based on https://github.com/scikit-image/scikit-image/blob/master/skimage/morphology/misc.py#L51
def remove_small_objects(out, min_size=64, connectivity=1, in_place=False):
    selem = ndi.generate_binary_structure(out.ndim, connectivity)
    ccs = np.zeros_like(out, dtype=np.int32)
    ndi.label(out, selem, output=ccs)
    component_sizes = np.bincount(ccs.ravel())
    too_small = component_sizes >= min_size
    return torch.Tensor(too_small[ccs]).byte()

# Methodology: 
# Neural Abstract Style Transfer for Chinese Traditional Painting
# https://arxiv.org/pdf/1812.03264.pdf
def mxdog(device, blur1, blur2, imgs, thres, gamma=1, epsilon=-1, phi=1):
    aux = xdog(blur1, blur2, imgs, gamma, epsilon, phi)
    mu = aux.mean((2, 3), True).expand_as(aux)
    aux = aux > mu
    mask = remove_small_objects(aux.to('cpu').numpy(), min_size=thres)
    return (aux * mask.to(device)).float()

def mxdog_loss(device, blur1, blur2, content_img, output_img, style_img, thres=64):
    N, C, H, W = output_img.shape
    
    def gram_matrix(matrix):
        tmp = matrix.view(-1, H, W)
        return torch.bmm(tmp, tmp.transpose(1,2)).view(N, C, H, H)
    
    I_md = mxdog(device, blur1, blur2, output_img, thres)
    I_c_md = mxdog(device, blur1, blur2, content_img, thres)
    I_s_md = mxdog(device, blur1, blur2, style_img, thres)
    
    content_loss = (output_img - I_c_md).norm() / output_img.numel()
    
    content_gram = gram_matrix(I_c_md)
    output_gram = gram_matrix(I_md)
    style_gram = gram_matrix(I_s_md)
    
    content_constraint_loss = (output_gram - content_gram).norm() / output_gram.numel()
    style_constraint_loss = (output_gram - style_gram).norm() / output_gram.numel()

    loss = 0.01 * content_loss + content_constraint_loss + style_constraint_loss
    return loss

class DoubleGatedGANModel(BaseModel):
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
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for auxilary loss')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for mxdog loss')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G', 'g', 'style', 'content', 'rec', 'tv', 'd_real', 'AC_style_real', 'AC_content_real', 'd_fake', 'AC_fake', 'AC_content_fake', 'mxdog']
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
            self.visual_names += ['rec']
            # self.visual_names += ['rec', 'content_only', 'style_only']
        else:
            self.visual_names += ['fake_B%d' % i for i in range(opt.n_style)]

        # define networks (both Generators and discriminators)
        self.netG_A = networks.define_Gated_G(opt.input_nc, opt.n_style, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.n_content)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.n_style, opt.n_content)

        if self.isTrain:
            self.style_autoflag = torch.zeros(opt.batch_size, opt.n_style + 1)
            self.style_autoflag[:, -1] = 1
            self.content_autoflag = torch.zeros(opt.batch_size, opt.n_content + 1)
            self.content_autoflag[:, -1] = 1
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
            sigma = 0.5 #todo move to option
            k = 1.6
            kernal_size = 3
            self.blur1 = networks.GaussianSmoothing(opt.input_nc, kernal_size, sigma).to(self.device)
            self.blur2 = networks.GaussianSmoothing(opt.input_nc, kernal_size, sigma * k).to(self.device)

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
        self.content_label_B = input['content_labels'].to(self.device)
        self.one_hot_label = torch.zeros(self.opt.batch_size, self.opt.n_style + 1).to(self.device)
        self.one_hot_label.scatter_(1, self.class_label_B.unsqueeze(1), 1)
        self.one_hot_content = torch.zeros(self.opt.batch_size, self.opt.n_content + 1).to(self.device)
        self.one_hot_content.scatter_(1, self.content_label_B.unsqueeze(1), 1)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A, self.one_hot_label, content_label=self.one_hot_content).to(self.device)  # G_A(A)
        if self.isTrain:
            self.rec = self.netG_A(self.real_A, self.style_autoflag, True, self.content_autoflag) # autoencoder
            # self.content_only = self.netG_A(self.real_A, self.style_autoflag, content_label=self.one_hot_content) # no style transformation
            # self.style_only = self.netG_A(self.real_A, self.one_hot_label, content_label=self.content_autoflag) # no content transformation
            assert self.rec.shape == self.real_A.shape
            # assert self.content_only.shape == self.real_A.shape
            # assert self.style_only.shape == self.real_A.shape
        else:
            for i in range(self.opt.n_style):
                flag = torch.zeros(self.opt.batch_size, self.opt.n_style + 1)
                flag[:, i] = 1
                setattr(self, 'fake_B%d' % i, self.netG_A(self.real_A, flag, content_label=self.one_hot_content))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        lambda_A = self.opt.lambda_A
        prediction, styles, contents = self.netD_A(real)
        self.loss_d_real = self.criterionGAN(prediction, True)
        N, _, H, W = styles.shape
        expanded_label = self.class_label_B.unsqueeze(1).unsqueeze(2).expand(N, H, W)
        self.loss_AC_style_real = lambda_A * self.criterionAC(styles, expanded_label)
        expanded_content = self.content_label_B.unsqueeze(1).unsqueeze(2).expand(N, H, W)
        self.loss_AC_content_real = lambda_A * self.criterionAC(contents, expanded_content)

        prediction, styles, contents = self.netD_A(fake)
        self.loss_d_fake = self.criterionGAN(prediction, False)
        self.loss_AC_fake = lambda_A * self.criterionAC(styles, expanded_label) # TODO: this was commented out in original implementation
        self.loss_AC_content_fake = lambda_A * self.criterionAC(contents, expanded_content)
        loss_D = self.loss_d_real + self.loss_d_fake \
            + (self.loss_AC_style_real + self.loss_AC_content_real + self.loss_AC_fake + self.loss_AC_content_fake)
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
        lambda_B = self.opt.lambda_B

        # auto-encoder loss
        self.loss_rec = autoencoder_constraint * self.criterionRec(self.rec, self.real_A)

        # gan loss
        prediction, styles, contents = self.netD_A(self.fake_B)
        # prediction, _, _ = self.netD_A(self.fake_B)
        # _, styles, _ = self.netD_A(self.style_only)
        # _, _, contents = self.netD_A(self.content_only)
        self.loss_g = self.criterionGAN(prediction, True)
        N, _, H, W = styles.shape
        expanded_label = self.class_label_B.unsqueeze(1).unsqueeze(2).expand(N, H, W)
        self.loss_style = self.criterionAC(styles, expanded_label)
        expanded_content = lambda_A * self.content_label_B.unsqueeze(1).unsqueeze(2).expand(N, H, W)
        self.loss_content = lambda_A * self.criterionAC(contents, expanded_content)
        
        # total variation loss
        if self.opt.tv_strength > 0:
            self.loss_tv = torch.sqrt(torch.mean((self.fake_B[:, :, :, :-1] - self.fake_B[:, :, :, 1:]) ** 2) 
            + torch.mean((self.fake_B[:, :, :-1, :] - self.fake_B[:, :, 1:, :]) ** 2))
        else:
            self.loss_tv = 0
        self.loss_tv *= self.opt.tv_strength
        
        self.loss_mxdog = lambda_B * mxdog_loss(self.device, self.blur1, self.blur2, self.real_A, self.fake_B, self.real_B)
        self.loss_G = self.loss_g + (self.loss_style + self.loss_content) + self.loss_rec + self.loss_tv + self.loss_mxdog
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
