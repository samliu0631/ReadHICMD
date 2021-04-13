import torch.nn as nn
import torch
from  addnetwork import  IdDis

class SimpleHICMD(nn.Module):
    def __init__(self, opt,hyperparameters):    
            # nn.Module的子类必须实现 __init__()  forward()
            # 这个函数中定义了HICMED模型中所有需要的层和属性。
            super(SimpleHICMD, self).__init__()
            # Setup the optimizers
            beta1    =  opt.id_dis_beta1        # Adam hyperparameter
            beta2    =  opt.id_dis_beta2        # Adam hyperparameter
            lr_id_d  =  opt.id_dis_lr_id_d      # Initial domain ID discriminator learning rate

            # init the domain discriminator.
            self.id_dis = IdDis(hyperparameters['gen']['id_dim'], hyperparameters['dis'], fp16=False)

            # list the parameters of domain discriminator.
            id_dis_params = list(self.id_dis.parameters())   # 将domain discirminator所有的参数列出来。

            # Set the optimizer of domain discrimiator   
            self.id_dis_opt = torch.optim.Adam([p for p in id_dis_params if p.requires_grad],
                                                lr=lr_id_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

