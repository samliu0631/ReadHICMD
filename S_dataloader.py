from torchvision.datasets import ImageFolder
import numpy as np
from operator import itemgetter
import math
from random import shuffle
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from S_data_sampler import S_PosNegSampler


# @Brief: 产生HICMDPP的训练数据加载器。
def get_mix_data_loaders(opt,config):
    
    # 获得数据的图像序号列表。
    idx_la, idx_lb = get_mix_image_index(config)
    # 获得训练数据的路径。
    datapath = config['data_root'] + '/train_all'
    
    # 获得
    dataloaders_a = GetMixDataLoader(opt, datapath, idx_la )
    dataloaders_b = GetMixDataLoader(opt, datapath, idx_lb )

    # 返回加载的数据。
    return dataloaders_a, dataloaders_b


def GetMixDataLoader(opt, datapath, idlist ):

    x = "train_all"
    # 用于对于图片数据的加载和预处理。
    transform_train_list = []  
    # 图像要插值成为256×128大小的图像。
    transform_train_list = transform_train_list + [transforms.Resize((opt.h,opt.w), interpolation=3)]  
    # opt.pad=0,所以不进行图像填充。
    transform_train_list = transform_train_list + [transforms.Pad(opt.pad)] if opt.pad > 0 else transform_train_list 
    # 控制随机裁剪图片。#  这里因为opt.pad=0，所以并不进行设置。
    transform_train_list = transform_train_list + [transforms.RandomCrop((opt.h,opt.w))] if opt.pad > 0 else transform_train_list  
    # 控制随机水平翻转
    transform_train_list = transform_train_list + [transforms.RandomHorizontalFlip()] if opt.flip else transform_train_list
    # 将数据转换为tensor
    transform_train_list = transform_train_list + [transforms.ToTensor()]  
    # 用于图像数据 均值和标准差的归一化
    transform_train_list = transform_train_list + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # initial the class transforms.Compose and form instance.
    data_transforms = transforms.Compose(transform_train_list)  
    # 
    image_datasets = S_PosNegSampler(datapath,idlist,data_transforms, data_flag = opt.data_flag,
                                              name_samping = opt.name_samping, num_pos = opt.samp_pos, num_neg = opt.samp_neg, opt=opt)
    
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.set_batchsize[x], shuffle=opt.set_shuffle[x],
                                                            num_workers=opt.set_workers[x], pin_memory=opt.pin_memory, drop_last=opt.set_droplast[x])
    
    return dataloaders




# @Brief: 获得源域和目标域图像加载的序号列表。
def get_mix_image_index(conf):
    
    # parameter initialization.
    bs = 1 
    batch_size = bs

    # 获得目录的路径。
    train_path  = conf['data_root']+'/train_all'
    mixData     = ImageFolder(train_path)
    ab_list     = mixData.imgs                       # 获得数据集合内的图像列表。
    ab_idx      = [i for i in range(len(ab_list))]   # 获得数据集合的图像总数。
    size_a      = conf['sample_a']                   # 源域的图像数量。
    size_b      = conf['sample_b']                   # 目标域的图像数量。

    # 两个数据集 图像序号的完整列表。
    a_full_idx = ab_idx[0:size_a]   # 获得源域的图像索引号。
    b_full_idx = ab_idx[size_a:]    # 获得目标域的图像索引号。

    # 确保两个数据集合使用同样数量的图像序号。
    if  size_a > size_b:
        sel_idx = list( np.random.choice(size_a, size_b , replace = False) ) # 从源域中抽取 和目标域一样数量的图像序号。
        a_idx   = list( itemgetter( *sel_idx )( a_full_idx  ) )              # 从a_full_idx中抽取 sel_idx 对应的序号。
        b_idx   = b_full_idx.copy()
    elif size_b > size_a:
        sel_idx = list( np.random.choice(size_b, size_a, replace = False ) )
        b_idx   = list( itemgetter( *sel_idx )( b_full_idx ) )
        a_idx = a_full_idx.copy()
    else:
        a_idx = a_full_idx.copy()
        b_idx = b_full_idx.copy()

    a_idx_a = a_idx.copy()  # 4120个源域图像索引号。
    a_idx_b = a_idx.copy()  # 4120个源域图像索引号。
    b_idx_a = b_idx.copy()  # 4120个目标域图像索引号。
    b_idx_b = b_idx.copy()  # 4120个目标域图像索引号。

    # generate two lists for train_loader_a and train_loader_b.
    ab_port = conf['ab_port']  # 0.1
    # 源域和目标域图像的共有数量。
    size_domain = min(size_a, size_b)         
    # ab_num: 每个域中和其他域图像进行 交叉训练的数量。           
    ab_num = math.floor(ab_port * size_domain) // bs * bs
    # xx_num: 每个域中 利用域内图像进行训练的数量。
    xx_num = (size_domain - ab_num) // bs * bs
    idx_la = []     # list for loader_a
    idx_lb = []     # list for loader_b

    # 从图像范围内随机选择 ab_num个图像的序号。 
    sel_idx_ab_a = list(np.random.choice(size_domain, ab_num, replace=False))   # 从图像范围内随机选择 ab_num个图像的序号。
    sel_idx_ab_b = list(np.random.choice(size_domain, ab_num, replace=False))   # 从图像范围中随机选择 ab_num个图像的序号。
    sel_idx_ba_a = list(np.random.choice(size_domain, ab_num, replace=False))   # 从图像范围中随机选择 ab_num个图像的序号。
    sel_idx_ba_b = list(np.random.choice(size_domain, ab_num, replace=False))   # 从图像范围中随机选择 ab_num个图像的序号。

    # 选择源域和目标域中 域内进行训练的图像序号。
    aa_idx_a = [a_idx_a[i] for i in range(size_domain) if i not in sel_idx_ab_a]  # batch aa for train_loader_a 
    aa_idx_b = [a_idx_b[i] for i in range(size_domain) if i not in sel_idx_ba_b]  # batch aa for train_loader_b  
    bb_idx_a = [b_idx_a[i] for i in range(size_domain) if i not in sel_idx_ba_a]  # batch bb for train_loader_a 
    bb_idx_b = [b_idx_b[i] for i in range(size_domain) if i not in sel_idx_ab_b]  # batch bb for train_loader_b  

    # 打乱顺序。
    shuffle(aa_idx_a)
    shuffle(aa_idx_b)
    shuffle(bb_idx_a)
    shuffle(bb_idx_b)

    aa_idx_a = aa_idx_a[:xx_num]  
    aa_idx_b = aa_idx_b[:xx_num] 
    bb_idx_a = bb_idx_a[:xx_num] 
    bb_idx_b = bb_idx_b[:xx_num]  

    # 获得源域和目标域交互训练的图像id号。
    ab_idx_a, ab_idx_b, ba_idx_a, ba_idx_b = [], [], [], []
    if sel_idx_ab_a != []:
        ab_idx_a = list(itemgetter(*sel_idx_ab_a)(a_idx_a))  # batch ab for train_loader_a    取值范围 0-12935       1280个编号。 图像在伪代码文件夹 中 对应的图形编号。
    if sel_idx_ab_b != []:
        ab_idx_b = list(itemgetter(*sel_idx_ab_b)(b_idx_b))  # batch ab for train_loader_b    取值范围 12936-25771   1280个编号。 图像在伪代码文件夹 中 对应的图形编号。
    if sel_idx_ba_a != []:
        ba_idx_a = list(itemgetter(*sel_idx_ba_a)(b_idx_a))  # batch ab for train_loader_a    取值范围 12936-25771   1280个编号。 图像在伪代码文件夹 中 对应的图形编号。
    if sel_idx_ba_b != []:
        ba_idx_b = list(itemgetter(*sel_idx_ba_b)(a_idx_b))  # batch ab for train_loader_b    取值范围 0-12935       1280个编号。 图像在伪代码文件夹 中 对应的图形编号。 

    # 进行序号的分配，确保两个序号之间相互对应。
    aa_thresh = conf['xx_port'] / 2               # 0.45
    bb_thresh = aa_thresh * 2                     # 0.9 
    ab_thresh = bb_thresh + conf['ab_port'] / 2   # 0.95
    while aa_idx_b or bb_idx_a or ab_idx_a or ba_idx_a: # 直到所有的列表元素都pop完毕之后。 11556,11556,1280,1280
        dice = np.random.uniform(0, 1)   # 从【0,1）中根据均匀分布随机采样
        if dice <= aa_thresh:
            if not aa_idx_a:
                continue
            for _ in range(batch_size):
                idx_la.append(aa_idx_a.pop())   # 表示训练数据有  45% 概率是 都来自于源域 数据集合。 （为了防止重复，从11556个编号中选择。）
                idx_lb.append(aa_idx_b.pop())
        elif dice > aa_thresh and dice <= bb_thresh:
            if not bb_idx_a:
                continue
            for _ in range(batch_size):
                idx_la.append(bb_idx_a.pop())  # 默认弹出列表中最后一个元素。     （为了防止重复，从11556个编号中选择。）
                idx_lb.append(bb_idx_b.pop())  # 训练数据有 45%的概率都来自目标域。
        elif dice > bb_thresh and dice <= ab_thresh:
            if not ab_idx_a:
                continue
            for _ in range(batch_size):
                idx_la.append(ab_idx_a.pop())   # 5% 的概率来自  源域 和 目标域（为了防止重复，从1280个编号中选择。）
                idx_lb.append(ab_idx_b.pop())
        else:
            if not ba_idx_a:
                continue
            for _ in range(batch_size):
                idx_la.append(ba_idx_a.pop())    # 5% 的概率来自  目标域 和 源域。（为了防止重复，从1280个编号中选择。）
                idx_lb.append(ba_idx_b.pop())

    return  idx_la, idx_lb
    




