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
pp = print_and_plot(opt)            # used to print and plot result.


# Set the parameters
config           = get_config("./Configs/RegDB2SYSU.yaml")
output_directory = "/home/tianyu/code/SamWorkSpace/HiCMD-master/data/MixDatasets"
epoch_round      = config['epoch_round_adv']
iterations       = 0 # 总的迭代次数。
nepoch           = 0 # 总的训练集训练次数。
   
# 跨域训练config['max_round']轮
for round_idx in range(config['max_round']): 
   
   # setup folders 
   round_output_directory = os.path.join(output_directory, str(round_idx)) # 确定每轮的路径，根据round编号更新路径。
   # 根据round基础路径 设置断点，图像，和伪标签的存储路径。
   checkpoint_directory, image_directory, pseudo_directory = prepare_sub_folder_pseudo(round_output_directory)  
   config['data_root'] = pseudo_directory  # 在配置config中 更新伪标签的存储路径。


   # At the round of adv_warm_max_round, we switch to self-training
   if round_idx == config['adv_warm_max_round']:   # config['adv_warm_max_round'] =1 
      config['lr2'] *= config['lr2_ramp_factor']   # The factor to multiply the lr2 after switching to self-training
      config['id_adv_w'] = 0.0         # 在自训练部分，停止进行域鉴别器的训练。
      config['id_tgt'] = True          # 在自训练部分，开始利用目标域的分类器结果计算loss。
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
      trainer.cnt_cumul = trainer.resume1(checkpoint_directory_prev)  # 不加载分类器，因为伪标签分类数量会变化。


   # 每个round产生数据集加载器。
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
         
         if order_a < config['ID_class_a'] and order_b < config['ID_class_a'] and config["aa"]:
            # print("aa")
            mode_code = "aa"
            trainer.go_train_mix(data_a, opt, 'train_all' , cnt, epoch_ridx, mode_code,config)  
            epoch_cnt = trainer.cnt_cumul  

         elif order_a >= config['ID_class_a'] and order_b >= config['ID_class_a'] and config["bb"]:
            # print('bb')
            mode_code = "bb"
            trainer.go_train_mix(data_b, opt, 'train_all' , cnt, epoch_ridx, mode_code,config ) 
            epoch_cnt = trainer.cnt_cumul
         
         elif order_a < config['ID_class_a'] and order_b >= config['ID_class_a'] and config["ab"]:
            # print("ab")
            mode_code ="ab"
            data_ab = GenerateMixData( data_a, data_b )
            trainer.go_train_mix(data_ab, opt, 'train_all' , cnt, epoch_ridx, mode_code, config ) 
            epoch_cnt = trainer.cnt_cumul


         elif order_a >= config['ID_class_a'] and order_b < config['ID_class_a']and config["ab"]:
            # print("ba")
            mode_code = "ba"
            data_ba = GenerateMixData( data_a, data_b ) # 仔细确认这部分。
            trainer.go_train_mix(data_ba, opt, 'train_all' , cnt, epoch_ridx, mode_code, config )
            epoch_cnt = trainer.cnt_cumul

         # Print, record, and plot the training results
         if ( (cnt + 1) % opt.cnt_print_loss == 0  ) :
            pp.print_info(opt, epoch_ridx, cnt, len(dataloaders_a) - 1, trainer.loss_type, trainer.acc_type, trainer.etc_type, trainer.cnt_cumul)  # 打印训练数据信息

         # print training index.
         iterations += 1
         subiterations += 1
         if (iterations + 1) % config['log_iter'] == 0:
            print("\033[1m Round: %02d  Epoch: %02d Iteration: %08d/%08d \033[0m \n" % (round_idx, nepoch, subiterations + 1, config['epoch_iteration'] * epoch_round), end=" ")

         # 如果累计训练达到1000次，进行一次存储。
         if epoch_cnt % 1000 == 0:      
            trainer.save(checkpoint_directory) # 存储到当前round路径的checkpoint路径。

         # 更新训练参数。
         trainer.update_learning_rate(opt, 'train_all') # 没此都要更新学习率。

      nepoch +=1  # update total epoch number.
      
      #################################################################
      # TODO: Add test module. and test.
      epoch_cnt = 5000
      opt.data_name = "SYSU"
      dataloaders, data_info = GenerateTestDataloader(opt)
      if epoch_cnt % opt.test_cnt == 0:
         trainer.eval()  # Set model to evaluate mode
         result, result_RAM, result_multi = extract_test_features(opt, trainer, dataloaders, data_info)
         for k in range(len(result)):
            result_k = result[k]
            save_path = save_test_features(opt, epoch_cnt, result_k, result_RAM, result_multi, k)
            CMC_single, ap_single = evaluate_result(opt, epoch_cnt, result_k, result_RAM, result_multi, save_path, k)
            pp.record_test_result(CMC_single, ap_single, epoch_cnt, k)
            pp.draw_and_save_info_test(opt, epoch_cnt, k)
         trainer.train()  # Set model to training mode


      ##############################################################    
      dataloaders_a, dataloaders_b = get_mix_data_loaders( opt, config )
      
      # adjust the total epochs per round
      epoch_ridx += 1  # update the epoch_ridx.
      if epoch_ridx == epoch_round and round_idx == 0:
         epoch_round = config['epoch_round']  #
         break

   trainer.save(checkpoint_directory)
   # 更新模型加载路径。
   checkpoint_directory_prev = checkpoint_directory        
         
      
