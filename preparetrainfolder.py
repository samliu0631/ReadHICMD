import os
from shutil import copyfile

download_path      = 'E:/WorkSpace/Data/HICMDDownload/data/RegDB_01'
Train_path         = download_path + '/train_all'
TrainRGB_save_path = download_path + '/trian_all_RGB'
if not os.path.isdir(TrainRGB_save_path):
    os.mkdir(TrainRGB_save_path)

for root, dirs, files in os.walk(Train_path, topdown=True):
    for name in files:
        if name[0]=='V':
            src_path = root + '/' + name
            dst_path = TrainRGB_save_path + '/' + name
            copyfile(src_path, dst_path)
        
  
