# -*- coding: utf-8 -*-

from __future__ import print_function, division
import matplotlib
matplotlib.use('agg')
from set_option import opt_model
from util_etc import *
from util_test import *
from util_train import *
from data_sampler import *
from reIDmodel_others import *
from trainer import HICMD
from collections import namedtuple

from HicmdPP import * # added by sam.
version =  torch.__version__


# set random seed
def set_seed(seed=0):
   rn.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
   # cudnn.enabled = False
   cudnn.deterministic = True
   cudnn.benchmark = False  


# 输入参数的定义
parser = argparse.ArgumentParser(description='Training')  # 创建ArgumentParser对象。
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0, 1, 2')
parser.add_argument('--flag_exp', default=1, type=int, help='1: original(1~2days), 0: for check (~1hour)')
parser.add_argument('--data_name',default='RegDB_01',type=str, help='RegDB_01 ~ RegDB_10 / SYSU')
parser.add_argument('--data_dir',default='./data/',type=str, help='data dir: e.g. ./data/')
parser.add_argument('--name_output',default='test', type=str, help='output name')
parser.add_argument('--test_only', default=False, type=bool, help='True / False')
parser.add_argument('--test_dir', default='./model/RegDB_01/test/', type=str, help='test_dir: e.g. ./path/')
parser.add_argument('--test_name', default='last', type=str, help='name of test: e.g. last')
parser.add_argument('--resume_dir', default='./model/RegDB_01/test/checkpoints/', type=str, help='resume_dir: e.g. ./path/checkpoints/')
parser.add_argument('--resume_name', default='', type=str, help='name of resume: e.g. last')

# 输入参数解析。
opt = parser.parse_args()  # parse the input arguments.

# 参数初始化
opt = opt_model(opt)  # Set the default parameters
np.random.seed(opt.random_seed) # Set the seed number of random algorithm
torch.manual_seed(opt.random_seed)  # 设置 (CPU) 生成随机数的种子.  目的是为了每次实验产生的随机数是一样的，有利于实验结果的比对。
torch.cuda.manual_seed_all(opt.random_seed)  # 为所有GPU设置随机数的种子。
opt = opt_settings(opt)



dataloaders, dataloaders_train_tsne, old_train_dataloader, data_info, data_sample, opt = data_settings(opt)
opt = opt_test_settings(opt)

trainer = HICMD(opt) # Initial the trainer
trainer.cnt_cumul = trainer.resume(opt)    

pseudo_label_generate(trainer,opt)