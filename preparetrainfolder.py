import os
from shutil import copyfile

#download_path      = 'E:/WorkSpace/Data/HICMDDownload/data/RegDB_01'
download_path      = '/home/tianyu/code/SamWorkSpace/HiCMD-master/data/SYSU'
Train_path         = download_path + '/train_all'
TrainRGB_save_path = download_path + '/trian_all_RGB'
TrainIR_save_path  = download_path + '/train_all_IR'

if not os.path.isdir(TrainRGB_save_path):
    os.mkdir(TrainRGB_save_path)
if not os.path.isdir(TrainIR_save_path):
    os.mkdir(TrainIR_save_path)



for root, dirs, files in os.walk(Train_path, topdown=True):
    for name in files:
        if name[0] == 'V':
            src_path = root + '/' + name
            partpath = root.split('/')
            dst_dir  = TrainRGB_save_path  + '/' +partpath[-1]
            dst_path = TrainRGB_save_path  + '/' +partpath[-1]+ '/' + name  
            if not os.path.isdir(dst_dir):
                os.mkdir(dst_dir)
            copyfile(src_path, dst_path)
        if name[0] == 'T':
            src_path = root + '/' + name
            partpath = root.split('/')
            dst_dir  = TrainIR_save_path  + '/' +partpath[-1]
            dst_path = TrainIR_save_path  + '/' +partpath[-1]+ '/' + name  
            if not os.path.isdir(dst_dir):
                os.mkdir(dst_dir)      
            copyfile(src_path, dst_path)
  
