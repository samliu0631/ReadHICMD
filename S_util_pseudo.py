import os
from shutil import rmtree
import torch
import numpy as np
from sklearn.cluster import DBSCAN , MiniBatchKMeans
from torchvision import datasets
from shutil import copyfile, copytree
from  re_ranking_one import re_ranking_one
from torchvision import transforms
from data_sampler import get_attribute



def prepare_sub_folder_pseudo(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    pseudo_directory = os.path.join(output_directory, 'pseudo_train')
    if not os.path.exists(pseudo_directory):
        print("Creating directory: {}".format(pseudo_directory))
        os.makedirs(pseudo_directory)
        os.makedirs(pseudo_directory + '/train_all')
    else:
        rmtree(pseudo_directory)
        os.makedirs(pseudo_directory)
        os.makedirs(pseudo_directory + '/train_all')

    return checkpoint_directory, image_directory, pseudo_directory



# 根据检测出的特征，根据分类数量，将特征进行分类和匹配。
def S_KmeansClusteringIR_RGB( target_features_IR,target_features_RGB, ClassNum ):
         # 对特征进行分组 clustering ##
        labels_IR , IR_Centers    = S_KmeansClustering(target_features_IR[0], ClassNum )
        labels_RGB , RGB_Centers  = S_KmeansClustering(target_features_RGB[0], ClassNum )
        RGBNum = int(max( labels_RGB ))  + 1 

        # Match the label        
        labels_RGB, labels_IR  = S_MatchingIRRGBLabels(labels_RGB, labels_IR , target_features_RGB, target_features_IR)
        return  labels_RGB, labels_IR


def S_ClusteringRGB_IR():
    pass



# @Brief: 根据伪标签将图像拷贝到对应的目录下。
def S_CopyImgsUsingPseudoLabels( image_datasets, n_samples_RGB, n_samples_IR, labels_RGB, labels_IR, config ):
    train_path_IR  = image_datasets['train_all_IR'].imgs
    train_path_RGB = image_datasets['train_all_RGB'].imgs 
    copy_save( labels_RGB, train_path_RGB, n_samples_RGB, config )  # 将目标域图像在伪标签文件夹下进行存储，同事
    copy_save( labels_IR,  train_path_IR,  n_samples_IR,  config )   
    copy_save_gt( config ) 
    return     
    
    



# @brief:根据特征的相似关系，来匹配红外和可见光之间的标签。
def S_MatchingIRRGBLabels(labels_RGB, labels_IR , target_features_RGB, target_features_IR):
    # # 记录分类后的图像标签类型数量。
    RGBNum = int(max( labels_RGB ))  + 1           
    IRNum  = int(max( labels_IR  ))  + 1        

    # 计算RGB图像， 对应分组的平均特征。
    RGBclassID      = range(RGBNum)
    RGBfeatureList  = []
    for i in range(RGBNum):
        FeatureIndexRGB = np.where(labels_RGB == i )
        print(FeatureIndexRGB)
        # 提取特征
        currentfeature = target_features_RGB[0][FeatureIndexRGB]
        #计算中值。
        meanfeature = torch.mean(currentfeature, dim = 0, keepdim = True)
        RGBfeatureList.append(meanfeature)

    # 计算IR图像中， 对应分组的平均特征。
    IRfeatureList = []
    for j in range(IRNum):
        FeatureIndexIR = np.where(labels_IR == j )
        currentFeature = target_features_IR[0][FeatureIndexIR]
        #　计算中值。
        meanfeature = torch.mean(currentFeature, dim = 0 , keepdim = True)
        IRfeatureList.append(meanfeature)

    # 根据分组数最小的分组，来计算匹配分类。
    if RGBNum > IRNum:
        MinimumClassNum = IRNum
        LargerClassNum  = RGBNum
        RefFeatureList  = IRfeatureList
        ComFeatureList  = RGBfeatureList
        Changedlabels   = labels_RGB
    else:
        MinimumClassNum = RGBNum
        LargerClassNum  = IRNum
        RefFeatureList  = RGBfeatureList
        ComFeatureList  = IRfeatureList
        Changedlabels   = labels_IR

    # 根据最小的分组，来进行分类的匹配。
    NewLabelList = [] 
    MinClassDist = [] 
    #UseFlag      = np.zeros(LargerClassNum)
    for i in range(MinimumClassNum):
        CurRef   = np.array( RefFeatureList[i] )
        MinDist  = 100000
        Index    = -1 
        for j in range(LargerClassNum):
            # 如果该类已经匹配过，就跳过。
            #if UseFlag[j]==1:
            #    continue
            CurCom   = np.array( ComFeatureList[j] )
            FeatDist = np.sqrt( np.sum( ( CurRef - CurCom ) **2 ) )
            if FeatDist < MinDist  :
                MinDist = FeatDist
                Index   = j

        # 记录对应的IR图像分类标签。
        MinClassDist.append( MinDist )
        NewLabelList.append( Index )
        #UseFlag[Index] = 1
    #
    Changedlabels = ChangeLabel(NewLabelList,Changedlabels)

    if RGBNum > IRNum:
        labels_RGB  = Changedlabels 
    else:
        labels_IR = Changedlabels

    # 返回更新后的标签结果。
    return  labels_RGB, labels_IR    




def ChangeLabel( NewLabelList, Changedlabels ):
    NewLabels = -1 * np.ones( len( Changedlabels ) )
    for i in range( len( NewLabelList ) ):
        Index = np.where(Changedlabels == NewLabelList[i] )
        NewLabels[Index] = i
    return NewLabels
            



# 对特征进行KMean分类。
def S_KmeansClustering(train_feature, ClusterNum):
    n_samples           = train_feature.shape[0]  # 获得特征的数量。
    kmeans              = MiniBatchKMeans(n_clusters = ClusterNum).fit(train_feature)
    labels              = kmeans.labels_
    kmeansCenter        = kmeans.cluster_centers_

    return labels, kmeansCenter




# 对特征进行DBSCAN分类。
def DBSCANclustering( train_feature):

    ######################################################################
    eps= 0.45
    min_samples= 7

    n_samples           = train_feature.shape[0]  # 获得特征的数量。
    train_feature_clone = train_feature.clone()
    train_dist          = torch.mm( train_feature_clone, torch.transpose(train_feature, 0, 1) )
    print(train_dist)

    print('--------------------------Start Re-ranking---------------------------')
    train_dist = re_ranking_one(train_dist.cpu().numpy())
    print('--------------------------Clustering---------------------------')
    # cluster

    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=8)
    ### non-negative clustering
    train_dist = np.maximum(train_dist, 0)
    ###
    cluster = cluster.fit(train_dist)
    print('Cluster Class Number:  %d' % len(np.unique(cluster.labels_)))
    # center = cluster.core_sample_indices_
    labels = cluster.labels_
    return labels

    


def copy_save( labels, train_path, n_samples, config ):
    ### copy pseudo-labels in target ###
    save_path = config['data_root'] + '/train_all'
    sample_b_valid = 0
    for i in range(n_samples):
        if labels[i] != -1:  # 剔除分配标签为-1的图像， 该图像在分类中被认为是野点。
            src_path = train_path[i][0]
            dst_id = labels[i]
            dst_path = save_path + '/' + 'B_' + str(int(dst_id))  # 目标域 分类后 的有标签文件夹路径。
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + os.path.basename(src_path))  # 将train_all中的图像对应拷贝到 伪标签文件夹中。
            sample_b_valid += 1
    config['sample_b'] = sample_b_valid  # 存储目标域中的有效图像数量。





def copy_save_gt( config):
        ### copy ground truth in source ###
    # train_all
    save_path = config['data_root'] + '/train_all'
    src_all_path = config['data_root_a']
    # for dukemtmc-reid, we do not need multi-query/
    src_train_all_path = os.path.join(src_all_path, 'train_all')
    subfolder_list = os.listdir(src_train_all_path)
    file_list = []
    for path, subdirs, files in os.walk(src_train_all_path):
        for name in files:
            file_list.append(os.path.join(path, name))
    config['ID_class_a'] = len(subfolder_list)    # 源域train_all文件夹下图像的类别。
    config['sample_a']   = len(file_list)         # 源域train_all文件夹下图像的总数量。
    for name in subfolder_list:
        copytree(src_train_all_path + '/' + name, save_path + '/A_' + name)   # 将源域train_all文件夹下的图像全部拷贝到伪标签代码下。
    
    return




# 为生成伪标签，而设置的数据集，数据记载器，数据集信息。
def GenerateDataloaderDataInfo( opt , Target_data_name ):
    transform_val_list = []  
    transform_val_list = transform_val_list + [transforms.Resize(size=(opt.h,opt.w),interpolation=3)]   
    transform_val_list = transform_val_list + [transforms.ToTensor()]
    transform_val_list = transform_val_list + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    data_transforms = {}          
    data_transforms['train_all']     = transforms.Compose(transform_val_list) 
    data_transforms['train_all_IR']  = transforms.Compose(transform_val_list)
    data_transforms['train_all_RGB'] = transforms.Compose(transform_val_list)

    datasetlist    = ['train_all','train_all_IR', 'train_all_RGB' ]
    image_datasets = { x: datasets.ImageFolder(os.path.join(opt.data_dir, Target_data_name, x),   data_transforms[x]     )   for x in datasetlist  }
    dataloaders    = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=False,
                                                    num_workers=16, pin_memory=True, drop_last=False) for x in datasetlist }
    data_info   = {}

    data_info['train_all_IR_cam'], data_info['train_all_IR_label'], \
        data_info['train_all_IR_modal'] = get_attribute( \
        opt.data_flag, image_datasets['train_all_IR'].imgs, flag = opt.type_domain_label)

    data_info['train_all_RGB_cam'], data_info['train_all_RGB_label'],  \
        data_info['train_all_RGB_modal'] = get_attribute( \
        opt.data_flag, image_datasets['train_all_RGB'].imgs, flag = opt.type_domain_label)

    return image_datasets, dataloaders , data_info




def  ExtractFeaturesFromTargetDomain( trainer, opt, dataloaders, data_info,Target_data_name):
    # extract features #################################################
    featurename_IR = "features_"+Target_data_name+"_IR.pt" # 用于保存目标域特征的文件名称。
    if os.path.exists(featurename_IR):
        target_features_IR = torch.load(featurename_IR)           
    else:           
        with torch.no_grad():
            target_features_IR = trainer.S_extract_features(opt, dataloaders['train_all_IR'], \
            data_info['train_all_IR_modal'], data_info['train_all_IR_cam'])   
        torch.save( target_features_IR, featurename_IR )

    featurename_RGB = "features_"+Target_data_name+"_RGB.pt" # 用于保存目标域特征的文件名称。
    if os.path.exists(featurename_RGB):
        target_features_RGB = torch.load(featurename_RGB)           
    else:           
        with torch.no_grad():
            target_features_RGB = trainer.S_extract_features(opt, dataloaders['train_all_RGB'], \
            data_info['train_all_RGB_modal'], data_info['train_all_RGB_cam'])   
        torch.save( target_features_RGB, featurename_RGB )

    return target_features_IR, target_features_RGB



def GeneratepPreKnownPseudoCode( config ): 
    
    # 将目标域图像 拷贝到伪标签目录下。
    save_path    = config['data_root'] + '/train_all'
    src_all_path = config['data_root_b']
    
    src_train_all_path = os.path.join(src_all_path, 'train_all')
    subfolder_list = os.listdir(src_train_all_path)
    file_list = []
    for path, subdirs, files in os.walk(src_train_all_path):
        for name in files:
            file_list.append(os.path.join(path, name))
    for name in subfolder_list:
        copytree(src_train_all_path + '/' + name, save_path + '/B_' + name) 
    config['sample_b']   = len(file_list)         # 源域train_all文件夹下图像的总数量。
    config['ID_class_b'] = len(subfolder_list)  # 这里直接进行了赋值。在真实情况下，是通过分类算法产生的。
    # 将源域图像 拷贝到伪标签目录下。 
    copy_save_gt( config )
    return