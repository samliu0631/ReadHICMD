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
import random as rn

from S_HicmdPP import *      # added by sam.
from S_dataloader import *   # added by sam.
from S_GetConfig import *    # added by sam.

version =  torch.__version__



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


# Set the parameters
config           = get_config("./Configs/RegDB2SYSU.yaml")
output_directory = "/home/tianyu/code/SamWorkSpace/HiCMD-master/data/MixDatasets"
epoch_round      = config['epoch_round_adv']

# 跨域训练config['max_round']轮
for round_idx in range(config['max_round']): 
   
   # setup folders 
   round_output_directory = os.path.join(output_directory, str(round_idx))
   # 设置断点，图像，和伪标签的存储路径。
   checkpoint_directory, image_directory, pseudo_directory = prepare_sub_folder_pseudo(round_output_directory)  
   config['data_root'] = pseudo_directory  # 存储伪标签的存储路径。


   # At the round of adv_warm_max_round, we switch to self-training
   if round_idx == config['adv_warm_max_round']:   # config['adv_warm_max_round'] =1 
      config['lr2'] *= config['lr2_ramp_factor']   # The factor to multiply the lr2 after switching to self-training
      config['id_adv_w'] = 0.0
      config['id_adv_w_max'] = 0.0     # TODO: 搞清楚这些参数的作用，以及是否需要使用。
      config['id_tgt'] = True          # TODO: 要考虑是否使用
      config['teacher'] = '' # we do not use teacher in the self-training
      if config['aa_drop']:
            config['aa'] = False

     
   if round_idx == 0:           
      config['ID_class_b'] = 0          # make sure class of target domain is 0 in initial round.    
      trainer = HICMDPP( opt, config )  # 
      trainer.cuda()  
      trainer.cnt_cumul = trainer.resume0(opt)   


   ### Pseudo-label generation ###
   GeneratepPreKnownPseudoCode( config )  # 这里用真实标签作为伪标签，目的在于测试模型是否能改善在目标域的测试结果。
   
   # TODO: Finish the pseudo label generation.
   # trainer.S_generate_pseudo_label(opt, config )  
   # TODO： forward 是否需要修改。 


   if round_idx > 0:           
      # Load the pretrained HICMD model.     
      trainer = HICMDPP( opt, config )   
      trainer.cuda()  
      trainer.cnt_cumul = trainer.resume1(checkpoint_directory_prev)  


   # 产生数据集加载器。
   dataloaders_a, dataloaders_b = get_mix_data_loaders(opt, config)
   print('Note that dataloader may hang with too much nworkers.')
   mixData_size = 2 * min(  config['sample_a'],  config['sample_b']  )
   config['epoch_iteration'] = mixData_size // config['batch_size']
   print( 'Every epoch need %d iterations' % config['epoch_iteration'] )

   # training
   subiterations = 0
   epoch_ridx = 0
   while epoch_ridx < epoch_round:  
      #for cnt, data in enumerate(dataloaders[phase])
      trainer.train()  # change to train mode.

      for cnt, ( data_a, data_b ) in enumerate( zip( dataloaders_a, dataloaders_b) ):  
         order_a = data_a[3]['order']
         order_b = data_b[3]['order']
         
         if order_a < config['ID_class_a'] and order_b < config['ID_class_a']:
            print("aa")
            mode_code = "aa"
            trainer.go_train_mix(data_a, opt, 'train_all' , cnt, epoch_ridx, mode_code,config)  
            epoch_cnt = trainer.cnt_cumul  

         elif order_a >= config['ID_class_a'] and order_b >= config['ID_class_a']:
            print('bb')
            mode_code = "bb"
            trainer.go_train_mix(data_b, opt, 'train_all' , cnt, epoch_ridx, mode_code,config ) 
            epoch_cnt = trainer.cnt_cumul
         
         elif order_a < config['ID_class_a'] and order_b >= config['ID_class_a']:
            print("ab")
            mode_code ="ab"
            data_ab = GenerateMixData( data_a, data_b )
            trainer.go_train_mix(data_ab, opt, 'train_all' , cnt, epoch_ridx, mode_code, config ) 
            epoch_cnt = trainer.cnt_cumul


         elif order_a >= config['ID_class_a'] and order_b < config['ID_class_a']:
            print("ba")
            mode_code = "ba"
            data_ba = GenerateMixData( data_a, data_b ) # 仔细确认这部分。
            trainer.go_train_mix(data_ba, opt, 'train_all' , cnt, epoch_ridx, mode_code, config )
            epoch_cnt = trainer.cnt_cumul


         # 更新训练参数。
         trainer.update_learning_rate(opt, 'train_all') 

      # TODO： 理清更新关系，确定config的更新参数。
      dataloaders_a, dataloaders_b = get_mix_data_loaders(opt, config)

   trainer.save(checkpoint_directory)
   # 更新模型加载路径。
   checkpoint_directory_prev = checkpoint_directory        
         
      
