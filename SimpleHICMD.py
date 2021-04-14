import torch.nn as nn
import torch
from  addnetwork import  IdDis

class SimpleHICMD(nn.Module):
    def __init__(self, opt,hyperparameters):    
            # nn.Module的子类必须实现 __init__()  forward()
            # 这个函数中定义了HICMED模型中所有需要的层和属性。
            super(SimpleHICMD, self).__init__()
            # Setup the optimizers.
            beta1    =  opt.id_dis_beta1        # Adam hyperparameter
            beta2    =  opt.id_dis_beta2        # Adam hyperparameter
            lr_id_d  =  opt.id_dis_lr_id_d      # Initial domain ID discriminator learning rate

            # init the domain discriminator.
            self.id_dis = IdDis(hyperparameters['gen']['id_dim'], hyperparameters['dis'], fp16=False)

            # list the parameters of domain discriminator.
            id_dis_params = list(self.id_dis.parameters())   # 将domain discirminator所有的参数列出来。

            # Set the optimizer of domain discriminator.   
            self.id_dis_opt = torch.optim.Adam([p for p in id_dis_params if p.requires_grad],
                                                lr=lr_id_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

            # Set the scheduler of the domain discriminator.
            self.id_dis_scheduler = get_scheduler(self.id_dis_opt, hyperparameters)

            self.id_dis_scheduler.gamma = hyperparameters['gamma2']

    def gen_update(self,hyperparameters):
        # clear the gradient.
        self.id_dis_opt.zero_grad()        
        #  what is fe_b? 
        #  fe_b 可以是原型编码   或者是    原型编码和风格属性编码的组合。  

        # calculate the loss of generator.  
        self.loss_gen_id_adv = self.id_dis.calc_gen_loss(fe_b) if hyperparameters['id_adv_w'] > 0 else 0


    def dis_update(self, x_a, x_b, hyperparameters):
        self.id_dis_opt.zero_grad()

        # model forward 
        # encode
        # x_a_single = self.single(x_a)
        #fe_a, fe_b是两个原型编码
        # print(x_ab)

        # Calculate loss
        self.loss_id_dis_aa, _, _ = self.id_dis.calc_dis_loss_ab(fe_a.detach(), fe_b.detach())
        self.loss_id_dis_total = hyperparameters['id_adv_w'] * self.loss_id_dis_aa
        self.loss_id_dis_total.backward()
        
        # check gradient norm
        self.loss_total_norm = 0.0
        for p in self.id_dis.parameters():
            param_norm = p.grad.data.norm(2)
            self.loss_total_norm += param_norm.item() ** 2
        self.loss_total_norm =  self.loss_total_norm ** (1. / 2)
        
        # optimize the parameter of the id-discriminator
        self.id_dis_opt.step()  