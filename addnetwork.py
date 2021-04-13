from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
#from torchvision import models
import torch.nn.init as init




def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun
##################################################################################
# Discriminator
##################################################################################

class IdDis(nn.Module):
    # Domain ID Discriminator architecture
    def __init__(self, input_dim, params, fp16):
        super(IdDis, self).__init__()
        self.n_layer = params['id_nLayer']    # number of layers in domain id discriminator
        self.dim = params['id_nFilter']       # number of layer filters in domain id discriminator 1024 
        self.gan_type = params['id_ganType']  # the type of the network of ID discriminator
        self.activ = params['id_activ']       # activation function style [relu/lrelu/prelu/selu/tanh] lrelu
        self.norm = params['id_norm']         # normalization layer [none/bn/in/ln]   bn
        self.ds = params['id_ds']             # down sampling rate in domain id discriminator   2 
        self.input_dim = input_dim            # 输入维度为appearance code的长度，  2024
        self.fp16 = fp16                      # fp16=False   这是一个标志量。用来描述是否使用了16位浮点数。

        self.fcnet = self.one_fcnet()         # 构建网络。
        self.fcnet.apply(weights_init('gaussian'))  # 这一步应该是给网络初始参数。

    def one_fcnet(self):
        dim = self.dim
        fcnet_x = []    # 对模型初始化。
        fcnet_x += [FcBlock(self.input_dim, dim, norm=self.norm, activation=self.activ)]
        # 增加初始的层。
        for i in range(self.n_layer - 2):
            # i从0开始，且初始层已经设置。 所以要剪掉2
            dim2 = max(dim // self.ds, 32)
            fcnet_x += [FcBlock(dim, dim2, norm=self.norm, activation=self.activ)]
            dim = dim2
        fcnet_x += [nn.Linear(dim, 1)]
        fcnet_x = nn.Sequential(*fcnet_x)
        return fcnet_x

    def forward(self, x):
        outputs = self.fcnet(x)
        outputs = torch.squeeze(outputs)
        return outputs

    def calc_dis_loss_ab(self, input_s, input_t):
        outs0 = self.forward(input_s)
        outs1 = self.forward(input_t)
        loss = 0

        reg = 0.0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2) # 0 indicates source and 1 indicates target
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss = loss+reg
        return loss, reg, 0.0

    def calc_dis_loss_aa(self, input_s0, input_s1):
        outs0 = self.forward(input_s0)
        outs1 = self.forward(input_s1)
        loss = 0

        reg = 0.0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 0)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.zeros_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss = loss+reg
        return loss, reg, 0.0

    def calc_dis_loss_bb(self, input_t0, input_t1):
        outs0 = self.forward(input_t0)
        outs1 = self.forward(input_t1)
        loss = 0

        reg = 0.0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss = loss+reg
        return loss, reg, 0.0

    def calc_gen_loss(self, input_t):
        outs0 = self.forward(input_t)
        loss = 0
        Drift = 0.001

        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0)**2) * 2  # LSGAN
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss
    




class FcBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, norm='none', activation='relu',fp16 = False):
        super(FcBlock, self).__init__()

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim, fp16 = fp16)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm1d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize fc layer
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor': # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x