import torch
from torchvision import datasets, transforms


def get_data_loader_folder(opt):
     # 用于对于图片数据的加载和预处理。

    # 用于训练
    transform_train_list = []  # List used to store the different class instances for training.
    transform_train_list = transform_train_list + [transforms.Resize((opt.h,opt.w), interpolation=3)]  # 图像要插值成为256×128大小的图像。
    transform_train_list = transform_train_list + [transforms.Pad(opt.pad)] if opt.pad > 0 else transform_train_list  # opt.pad=0,所以不进行图像填充。
    transform_train_list = transform_train_list + [transforms.RandomCrop((opt.h,opt.w))] if opt.pad > 0 else transform_train_list  # 控制随机裁剪图片。#  这里因为opt.pad=0，所以并不进行设置。
    transform_train_list = transform_train_list + [transforms.RandomHorizontalFlip()] if opt.flip else transform_train_list# 控制随机水平翻转
    transform_train_list = transform_train_list + [transforms.ToTensor()]  # 将数据转换为tensor，并归一化至【0,1】范围内。
    transform_train_list = transform_train_list + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    # 用于测试
    transform_val_list = []  # List used to store the different class instances for evaluation.
    transform_val_list = transform_val_list + [transforms.Resize(size=(opt.h,opt.w),interpolation=3)]   # calling __init__ of each class.
    transform_val_list = transform_val_list + [transforms.ToTensor()]
    transform_val_list = transform_val_list + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    # 生成datatransform对象
    data_transforms = {}            # Initial dict.
    for x in opt.phase_data:        # 'train_all', 'gallery', 'query'
        if x == opt.phase_train:    # if x == 'train_all'
            data_transforms[x] = transforms.Compose(transform_train_list)  # initial the class transforms.Compose and form instance.
        else:
            data_transforms[x] = transforms.Compose(transform_val_list)    # initial transforms.Compose class.
   

    image_datasets = { x: datasets.ImageFolder(    os.path.join(opt.data_dir, opt.data_name, x),   data_transforms[x]     )   for x in opt.phase_data}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.set_batchsize[x], shuffle=opt.set_shuffle[x],
                                                  num_workers=opt.set_workers[x], pin_memory=opt.pin_memory, drop_last=opt.set_droplast[x]) for x in opt.phase_data}
    return dataloaders


def pseudo_label_generate(trainer,opt):
        #
        ### Feature extraction ###
        trainer.eval()   #  注意这里使用的是训练模型来提取特征。
        test_dir = opt['data_root_b']   # data/DukeMTMC/pytorch
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(( opt.h,opt.w ), interpolation=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # data_dir = test_dir
        image_datasets = { x: datasets.ImageFolder(    os.path.join(opt.data_dir, opt.data_name, x),   data_transforms     )   for x in opt.phase_data}
    
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.set_batchsize[x], shuffle=opt.set_shuffle[x],
                                                  num_workers=opt.set_workers[x], pin_memory=opt.pin_memory, drop_last=opt.set_droplast[x]) for x in opt.phase_data}
    
        # image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['train_all']}
        # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt['test_batchsize'],
        #                                               shuffle=False, num_workers=0) for x in ['train_all']}
       
       
       
        train_path = image_datasets['train_all'].imgs

        # Extract feature
        with torch.no_grad():
            train_feature = trainer.extract_feature(dataloaders['train_all'], opt) # 提取目标数据集合上图像的特征。 16522* 1024。 16522张图，特征维度为1024.
        trainer.train()

        ### clustering ###
        labels = self.clustering(train_feature, train_path, opt)   # 根据训练结果，对特征进行分类，并贴上分类后的标签。  16522维的array

        ### copy and save images ###
        n_samples = train_feature.shape[0]                  # 获得目标域训练图像的数量。
        opt['ID_class_b'] = int(max(labels)) + 1            # 记录分类后的图像标签类型数量。
        self.copy_save(labels, train_path, n_samples, opt)  # 将目标域图像在伪标签文件夹下进行存储，同事
        return