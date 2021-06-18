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
from util_pseudo import * # added by sam.

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
opt = opt_model(opt)                         # Set the default parameters
np.random.seed(opt.random_seed)              # Set the seed number of random algorithm
torch.manual_seed(opt.random_seed)           # 设置 (CPU) 生成随机数的种子.  目的是为了每次实验产生的随机数是一样的，有利于实验结果的比对。
torch.cuda.manual_seed_all(opt.random_seed)  # 为所有GPU设置随机数的种子。
opt = opt_settings(opt)

dataloaders_Source, dataloaders_train_tsne_Source, old_train_dataloader_Source, data_info_Source, data_sample_Source, opt = data_settings(opt)
opt = opt_test_settings(opt)


# 将目标训练集合中的数据按照红外和可见光进行分类。





# Set the parameters
config                       = {}
config['max_round']          = 40
config['adv_warm_max_round'] = 1
config['lr2_ramp_factor']    = 60
config['aa_drop']            = True     # Whether to drop the aa batch in self-training
config['data_root_a']        = "/home/tianyu/code/SamWorkSpace/HiCMD-master/data/RegDB_01"
output_directory             = "/home/tianyu/code/SamWorkSpace/HiCMD-master/data/MixDatasets"

for round_idx in range(config['max_round']): # config['max_round']=40   40轮
   ### setup folders
   round_output_directory = os.path.join(output_directory, str(round_idx))
   checkpoint_directory, image_directory, pseudo_directory = prepare_sub_folder_pseudo(round_output_directory)  # 设置断点，图像，和伪标签的存储路径。
   config['data_root'] = pseudo_directory  # 存储伪标签的存储路径。

   # In the initial round, we disenable self-training and warmup the network with adversarial training
   # At the round of adv_warm_max_round, we switch to self-training
   if round_idx == config['adv_warm_max_round']:   # config['adv_warm_max_round'] =1 
      config['lr2'] *= config['lr2_ramp_factor']   # The factor to multiply the lr2 after switching to self-training
      config['id_adv_w'] = 0.0
      config['id_adv_w_max'] = 0.0
      config['id_tgt'] = True
      config['teacher'] = '' # we do not use teacher in the self-training
      if config['aa_drop']:
            config['aa'] = False

   ### Evaluate source model ###
   if round_idx == 0: # 如果是第一轮训练，就需要加载预训练的模型。
      ### Model initialization with source model for test ###
      # Load the pretrained HICMD model.
      trainer = HICMDPP(opt) 
      trainer.cnt_cumul = trainer.resume(opt)    
      trainer.cuda()  # 将模型移动到cuda上。
      # 读取在源域数据集合上，预先训练好的DGNet模型。
      # trainer.test(config)   # annotated by sam 
      # #write_loss(iterations, trainer, train_writer)  # annotated by sam 
      # rank1 = trainer.rank_1
      # rank5 = trainer.rank_5
      # rank10 = trainer.rank_10
      # mAP0 = trainer.mAP_zero
      # mAP05 = trainer.mAP_half
      # mAPn1 = trainer.mAP_neg_one

      # mAP_list.append(mAP05)
      # rank1_list.append(rank1.numpy())
      # rank5_list.append(rank5.numpy())
      # rank10_list.append(rank10.numpy())

   ### Pseudo-label generation ###
   trainer.generate_pseudo_label(opt,config)  # 在这里进行Pseudo-label的生成。这个是DGNet没有的。
   #　目标域的训练图像就是train_all文件夹下面的所有图像。只是在使用的时候不利用文件夹名称的标签信息。而是通过特征聚类来分配标签。
