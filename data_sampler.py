from torchvision import datasets
import os
import numpy as np
import random
import torch

class PosNegSampler(datasets.ImageFolder):

    def __init__(self, root, transform, data_flag = 1, name_samping = 'RAND', num_pos = 4, num_neg = 0, opt = ''):
        super(PosNegSampler, self).__init__(root, transform)
        self.cams, self.real_labels, self.modals = get_attribute(data_flag, self.samples, flag = 0)
        # 获得数据集合中每张图像 对应的标签 和模态编号。


        self.num_pos = num_pos  # 表示正向实例数量
        self.num_neg = num_neg  # 表示负实例数量。
        self.name_samping = name_samping  # ['P_PAIR', 'N_PAIR']
        self.opt = opt

    def _get_cam_id(self, path):
        camera_id = []
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        #camera_id = filename.split('_')[2][0:2]
        return int(camera_id)-1

    def _get_pair_pos_sample(self, index):
        # 找到和index 同一个类的图像 序号（　剔除index　）###################################################
        pos_index = np.argwhere(np.asarray(self.real_labels) == np.asarray(self.real_labels[index]))
        pos_index = pos_index.flatten()     # 降到1维。
        pos_index = np.setdiff1d(pos_index, index) # delete index 删除当前index图像。

        # 根据pos_index找到和index同类不同域的图像序号cross_index ##################################################
        modal = self.modals[index] #　当前图像对应的模态。
        cross_index = []
        for i in range(len(pos_index)):
            if modal != self.modals[pos_index[i]]:#如果当前图像和index对应图像 不同域。
                cross_index.append(pos_index[i]) # 记录同一个类，但是不同域的图像ID.

        # 记录index同类 的红外中心图像序号 IR_pivot_idx， 可见光中心图像序号 RGB_pivot_idx ###################################
        flag_IR_pos_same_cam = False
        if modal == 0: # 0:IR
            IR_pivot_idx  = index # 记录和index同类的 红外轴图像序号
            # 记录和index 同类的RGB图像序号
            RGB_pivot_idx = int( cross_index[np.random.permutation( len(cross_index) )[0]] ) #　记录可见光的轴图像
        else:
            IR_pivot_idx  = int(cross_index[np.random.permutation(len(cross_index))[0]])  # 记录红外的轴图像。
            RGB_pivot_idx = index  # 记录可见光的周图像。

        # IR_pivot_cam：红外图像的相机
        IR_pivot_cam = self.cams[IR_pivot_idx] # 红外轴图像的相机类型
        # IR_same_cam_idx：所有红外图像的索引号。
        IR_same_cam_idx = np.argwhere(np.asarray(self.modals) == np.asarray(self.modals[IR_pivot_idx])).flatten()
        # IR_same_cam_idx：打乱顺序的红外图像索引号。
        IR_same_cam_idx = IR_same_cam_idx[np.random.permutation(len(IR_same_cam_idx))]
        # IR_pivot_cam = self.cams[IR_pivot_idx]

        # other IR pivot (same cam, different ID)# 找到除了红外中心图像外的 另一个红外中心图像，不同类。
        IR_pivot_idx_all = []# 用来记录红外中心图像
        IR_pivot_label_all = []# 记录红外中心图像的类别索引
        IR_pivot_idx_all.append(IR_pivot_idx)  # 在IR_pivot_idx_all中添加红外轴图像序号。
        IR_pivot_label_all.append(self.real_labels[IR_pivot_idx]) #　添加红外轴图像的类别号．
        cnt = 0
        is_find = False# self.opt.pos_mini_batch =2  表示正向图像的最小batch=2
        if self.opt.pos_mini_batch > 1: # 从数据库中选择　和中心图像　不同类但同域的　1个标签　作为轴图像．
            while not is_find:
                selected_idx = int( IR_same_cam_idx[cnt] )  # 选择一张红外图像
                if not self.real_labels[selected_idx] in IR_pivot_label_all: # 如果当前红外图像和中心红外图像不是一类。
                    IR_pivot_idx_all.append(selected_idx)   # 记录当前和index不同类的红外图像
                    IR_pivot_label_all.append(self.real_labels[selected_idx])  # 记录红外中心图像的标签．
                if len(IR_pivot_idx_all) == self.opt.pos_mini_batch: #只有两个，两个红外不同类图像，包括index所在类．
                    is_find = True
                cnt += 1

        # find IR pos/neg sample
        selected_pos_index = []
        selected_pos_path = []
        for k in range(len(IR_pivot_idx_all)): 
            # IR_pivot_idx_all中，有一个和index同类的红外图像，有一个和index不同类的红外图像。
            #　遍历2个不同类的红外轴图像。
            one_set = []
            selected_idx = IR_pivot_idx_all[k]    # 当前红外轴图像．
            # 获得当前红外中心图像 同类的 图像索引（包括IR和RGB）
            pos_index = np.argwhere(np.asarray(self.real_labels) == np.asarray(self.real_labels[IR_pivot_idx_all[k]])).flatten()
            #　剔除index自身。
            pos_index = np.setdiff1d(pos_index, index)
            # 打乱图像索引顺序
            pos_index = pos_index[np.random.permutation(len(pos_index))] # 得到和中心图像不同类的图像编号。
            if_find = False
            cnt = 0 
            cnt_yes = 0
            pos_same_cam = [IR_pivot_idx_all[k]]# 记录当前红外轴图像序号。
            if k == 0:
                pos_diff_modal = [RGB_pivot_idx]# 记录和index同类的RGB图像。
            else:
                pos_diff_modal = []
            while cnt_yes != 2:
                # 判断当前红外轴图像　和　相同类的图像（IR和RGB）　是红外图形还是可见光图像。

                # 如果当前图像是RGB图像。 opt.samp_pos = 2
                if (self.modals[pos_index[cnt]] != self.modals[selected_idx]): # diff modal
                    if len(pos_diff_modal) < self.opt.samp_pos:
                        pos_diff_modal.append(int(pos_index[cnt]))# 记录同类的RGB图像序号。
                        if len(pos_diff_modal) == self.opt.samp_pos:
                            # self.opt.samp_pos: 表示正向图像的采样数量。=2
                            cnt_yes += 1
                # 如果当前图像是RGB图像。
                elif self.modals[pos_index[cnt]] == self.modals[selected_idx]: # same modal
                    if len(pos_same_cam) < self.opt.samp_pos:
                        pos_same_cam.append(int(pos_index[cnt]))# 记录同类的IR图像序号。
                        if len(pos_same_cam) == self.opt.samp_pos:
                            cnt_yes += 1
                cnt += 1
            one_set.extend(pos_diff_modal)# 记录同类RGB图像序号。
            one_set.extend(pos_same_cam)# 记录同类IR图像序号。
            selected_pos_index.extend(one_set)# 记录同类的IR,RGB图像序号。共计8张图像。
            #分别是index类对应的 2个同类RGB图像，2个同类IR图像。  非index类对应的2个同类RGB图象，2个同类IR图像。
            #前4张图是对应index类别的，后4张图不对应index类别。
        for i in range(len(selected_pos_index)):
            selected_pos_path.append(self.samples[selected_pos_index[i]][0])#记录每个图像的路径。
        # for i in range(len(selected_pos_index)):
        #     print('modal: {}, ID: {}, cam: {}'.format(self.modals[selected_pos_index[i]], self.real_labels[selected_pos_index[i]],
        #                                             self.cams[selected_pos_index[i]]))

        return selected_pos_path, selected_pos_index



    def _get_pos_sample(self, index):

        pos_index = np.argwhere(np.asarray(self.real_labels) == np.asarray(self.real_labels[index]))
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index) # delete index
        # same label: pos_index

        modal = self.modals[index]
        cam = self.cams[index]
        mono_index = []
        cross_index = []
        for i in range(len(pos_index)):
            selected_index = pos_index[i]
            selected_modal = self.modals[selected_index]
            if modal == selected_modal:
                mono_index.append(selected_index)
            else:
                cross_index.append(selected_index)


        if 'P_RAND' in self.name_samping: # [n]
            num_mono = self.num_pos
            num_cross = 0
            mono_index = pos_index
        elif 'P_MONO' in self.name_samping: # [n/0]
            num_mono = self.num_pos
            num_cross = 0
        elif 'P_CROSS' in self.name_samping: # [0/n]
            num_mono = 0
            num_cross = self.num_pos
        elif 'P_MULTI1' in self.name_samping: # [n/1] (n>=2)
            num_mono = self.num_pos - 1
            num_cross = 1
        elif 'P_MULTI2' in self.name_samping: # [1/n] (n>=2)
            num_mono = 1
            num_cross = self.num_pos - 1

        if (num_mono < 0) or (num_cross < 0):
            print('please check sampling num_pos')
            assert False


        rand_mono = np.random.permutation(len(mono_index))
        selected_pos_path = []
        selected_pos_index = []
        for i in range(num_mono):
           t = i % len(rand_mono)
           tmp_index = mono_index[rand_mono[t]]
           selected_pos_index.append(tmp_index)
           selected_pos_path.append(self.samples[tmp_index][0])

        rand_cross = np.random.permutation(len(cross_index))
        for i in range(num_cross):
           t = i % len(rand_cross)
           tmp_index = cross_index[rand_cross[t]]
           selected_pos_index.append(tmp_index)
           selected_pos_path.append(self.samples[tmp_index][0])


        return selected_pos_path, selected_pos_index

    def _get_pair_neg_sample(self, pos_label, pos_cam):

        used_label = list(set(pos_label.tolist()))
        rand_idx = np.random.permutation(len(self.real_labels))

        IR_idx_all = []
        RGB_idx_all = []
        cnt = 0

        is_find = False
        while not is_find:
            selected_idx = int(rand_idx[cnt])
            selected_label = self.real_labels[selected_idx]
            if not self.real_labels[selected_idx] in used_label:

                if self.modals[selected_idx] == 1: # RGB
                    if len(RGB_idx_all) < self.opt.neg_mini_batch:
                        RGB_idx_all.append(selected_idx)
                        used_label.append(selected_label)
                elif self.modals[selected_idx] == 0: # IR
                    if len(IR_idx_all) < self.opt.neg_mini_batch:
                        IR_idx_all.append(selected_idx)
                        used_label.append(selected_label)

            if (len(RGB_idx_all) == self.opt.neg_mini_batch) and (len(IR_idx_all) == self.opt.neg_mini_batch):
                is_find = True
            cnt += 1
        selected_neg_index = []
        selected_neg_index.extend(RGB_idx_all)
        selected_neg_index.extend(IR_idx_all)

        selected_neg_path = []
        for i in range(len(selected_neg_index)):
            selected_neg_path.append(self.samples[selected_neg_index[i]][0])
        # for i in range(len(selected_neg_index)):
        #     print('modal: {}, ID: {}, cam: {}'.format(self.modals[selected_neg_index[i]], self.real_labels[selected_neg_index[i]],
        #                                             self.cams[selected_neg_index[i]]))

        return selected_neg_path, selected_neg_index


    def _get_neg_sample(self, index):

        neg_index = np.argwhere(np.asarray(self.real_labels) != np.asarray(self.real_labels[index]))
        neg_index = neg_index.flatten()

        modal = self.modals[index]
        mono_index = []
        cross_index = []
        for i in range(len(neg_index)):
            selected_index = neg_index[i]
            selected_modal = self.modals[selected_index]
            if modal == selected_modal:
                mono_index.append(selected_index)
            else:
                cross_index.append(selected_index)

        if 'N_RAND' in self.name_samping: # [n]
            num_mono = self.num_neg
            num_cross = 0
            mono_index = neg_index
        elif 'N_MONO' in self.name_samping: # [n/0]
            num_mono = self.num_neg
            num_cross = 0
        elif 'N_CROSS' in self.name_samping: # [0/n]
            num_mono = 0
            num_cross = self.num_neg
        elif 'N_MULTI1' in self.name_samping: # [n/1] (n>=2)
            num_mono = self.num_neg - 1
            num_cross = 1
        elif 'N_MULTI2' in self.name_samping: # [1/n] (n>=2)
            num_mono = 1
            num_cross = self.num_neg - 1

        if (num_mono < 0) or (num_cross < 0):
            print('please check sampling num_neg')
            assert False

        rand_mono = np.random.permutation(len(mono_index))
        selected_neg_path = []
        selected_neg_index = []
        for i in range(num_mono):
            t = i % len(rand_mono)
            tmp_index = mono_index[rand_mono[t]]
            selected_neg_index.append(tmp_index)
            selected_neg_path.append(self.samples[tmp_index][0])

        rand_cross = np.random.permutation(len(cross_index))
        for i in range(num_cross):
            t = i % len(rand_cross)
            tmp_index = cross_index[rand_cross[t]]
            selected_neg_index.append(tmp_index)
            selected_neg_path.append(self.samples[tmp_index][0])

        return selected_neg_path, selected_neg_index



    def __getitem__(self, index):       # 这个函数定义了如何对数据集合进行数据的调用
        ori_path, order = self.samples[index]  #　这里的order应该是按照顺序索引文件夹　得到的序号。
        real_label = self.real_labels[index]   # 获得图片的真实标签。
        cam = self.cams[index]#获得相机的编号。
        modal = self.modals[index]# 获得图像模态编号。
        attribute = {'order':order, 'label':real_label, 'cam':cam, 'modal':modal}
        ori = self.loader(ori_path)  # 加载图像。 PIL格式。
        if self.transform is not None:
            ori = self.transform(ori)  # 将图像转换为tensor格式。

        # self.num_pos= opt.samp_pos = 2
        if self.num_pos > 0:
            if 'P_PAIR' in self.name_samping:
                pos_path, pos_index = self._get_pair_pos_sample(index)
                # 获得index所在类的2张RGB图像和2张红外图像。以及和index不同类的2张RGB图像和2张红外图像。
            else:
                pos_path, pos_index = self._get_pos_sample(index)
            pos_cam = []
            pos_modal = []
            pos_order = []
            pos_label = []
            for i in range(len(pos_index)):
                # 遍历8张图像的索引号。
                pos_cam.append(self.cams[pos_index[i]])  # 获得图像对应的相机类型
                pos_modal.append(self.modals[pos_index[i]]) #获得图像对应的模态类型
                pos_order.append(self.samples[pos_index[i]][1]) # 获得图像对应类别索引
                pos_label.append(self.real_labels[pos_index[i]])# 获取图像的标签号

            pos_image = [0 for _ in range(len(pos_index))]
            for i in range(len(pos_index)):
                pos_image[i] = self.loader(pos_path[i])# 加载图像。

            if self.transform is not None:
                for i in range(len(pos_index)):
                    pos_image[i] = self.transform(pos_image[i]) #对图像进行变换，转换为tensor类型。

            if self.target_transform is not None:
                pass
                # label_t = self.target_transform(label_t)

            c,h,w = pos_image[0].shape # 获取图像的通道数，高度，宽度。
            for i in range(len(pos_index)):
                pos_image[i] = pos_image[i].view(1,c,h,w) # view()函数用来调整图像的维度。
            pos = pos_image[0]
            # 将8个tensor变量整合到一个tensor中。
            for i in range(len(pos_index)-1):
                pos = torch.cat((pos, pos_image[i+1]), 0) # 将tensor按行进行拼接。
            pos_order = torch.as_tensor(pos_order) # 将变量类型转换为tensor
            pos_label = torch.as_tensor(pos_label)
            pos_cam = torch.as_tensor(pos_cam)
            pos_modal = torch.as_tensor(pos_modal)
            # 生成字典数据。
            attribute_pos = {'order':pos_order, 'label':pos_label, 'cam':pos_cam, 'modal':pos_modal}
        else:
            pos = []
            attribute_pos = {}



        # opt.neg_mini_batch 
        # self.num_neg = opt.samp_neg = 1 
        if self.num_neg > 0:
            if 'N_PAIR' in self.name_samping:
                neg_path, neg_index = self._get_pair_neg_sample(pos_label, pos_cam[self.opt.samp_pos].item())
            else:
                neg_path, neg_index = self._get_neg_sample(index)

            neg_cam = []
            neg_modal = []
            neg_order = []
            neg_label = []
            for i in range(len(neg_index)):
                neg_cam.append(self.cams[neg_index[i]])
                neg_modal.append(self.modals[neg_index[i]])
                neg_order.append(self.samples[neg_index[i]][1])
                neg_label.append(self.real_labels[neg_index[i]])

            neg_image = [0 for _ in range(len(neg_index))]
            for i in range(len(neg_index)):
                neg_image[i] = self.loader(neg_path[i])

            if self.transform is not None:
                for i in range(len(neg_index)):
                    neg_image[i] = self.transform(neg_image[i])

            if self.target_transform is not None:
                pass
                # label_t = self.target_transform(label_t)

            c,h,w = neg_image[0].shape
            for i in range(len(neg_index)):
                neg_image[i] = neg_image[i].view(1,c,h,w)
            neg = neg_image[0]
            for i in range(len(neg_index)-1):
                neg = torch.cat((neg, neg_image[i+1]), 0)
            neg_order = torch.as_tensor(neg_order)
            neg_label = torch.as_tensor(neg_label)
            neg_cam = torch.as_tensor(neg_cam)
            neg_modal = torch.as_tensor(neg_modal)
            attribute_neg = {'order':neg_order, 'label':neg_label, 'cam':neg_cam, 'modal':neg_modal}
        else:
            neg = []
            attribute_neg = {}

        # pos = torch.cat((pos0.view(1,c,h,w), pos1.view(1,c,h,w), pos2.view(1,c,h,w), pos3.view(1,c,h,w)), 0)
        return ori, pos, neg, attribute, attribute_pos, attribute_neg


def get_attribute(data_flag, img_samples, flag):
    # data_flag:   a  number，用来表示对那个数据集合进行操作。
    # img_samples:  list all the path and type of images.
    # flag: number，

    cams = []
    labels = []
    modals = []
    for path, idx in img_samples:
        # iterate the path and type id of each image.
        labels.append(get_real_label(path, data_flag))  # 获得图片所在类的编号，就是图片所在文件夹的命名编号。
        cams.append(get_cam(path, data_flag))     # 获得相机的属性编码， 0表示红外相机，1表示可见光相机。
        modals.append(gel_modal(path, data_flag))  # 获取图像的模态编号， 0表示红外图像，1 表示可见光图像。

    cams = np.asarray(cams)   # 将相机的属性编码，变换为ndarray数据类型。

    if flag == 1:  # [1,2,4,5]->0, 3->1, 6->2
        change_set = [[1, 0], [2, 0], [4, 0], [5, 0], [3, 1], [6, 2]]
    elif flag == 2:  # [1,2]->0, [4,5]->1, 3->2, 6->3
        change_set = [[1, 0], [2, 0], [4, 1], [5, 1], [3, 2], [6, 3]]
    elif flag == 3:  # X = X - 1
        change_set = [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]]
    elif flag == 0:  # [1,2,4,5]->1, [3,6]->0  
        change_set = [[1, 1], [2, 1], [3, 0], [4, 1], [5, 1], [6, 0]]
    for i in range(len(change_set)): # 如下代码，实际使用并没有起到实际作用。
        cams[np.where(cams == change_set[i][0])[0]] = int(change_set[i][1])  # 把多余的相机的编号进行赋值。
    cams = list(cams)

    return cams, labels, modals

def get_cam(path, flag):# 通过图像的名称来判断图像的类型，即红外还是可见光。
    filename = os.path.basename(path)
    if flag == 1:  # Market1501
        return int(filename.split('c')[1][0])
    elif flag == 5: # RegDB
        if filename[0] == 'T':  # Thermal : 0
            return int(0)
        else:
            return int(1)
    elif flag == 6: # SYSU
        return int(filename[filename.find('cam')+3])

def get_real_label(path, flag):
    filename = os.path.basename(path)
    if flag == 1:  # Market1501
        label = filename[0:4]
        if label[0:2] == '-1':
            return int(-1)
        else:
            return int(label)
    elif flag == 5: # RegDB
        return int(path.split('/')[-2])   # 将图片所在文件夹的编号，作为分类标签。
    elif flag == 6: # SYSU
        return int(path.split('/')[-2])


def gel_modal(path, flag):
    filename = os.path.basename(path)
    if flag == 1:  # Market1501
        return int(0)
    elif flag == 5: # RegDB
        if filename[0] == 'T':  # Thermal : 0
            return int(0)       # 根据图像首字符来判断图像的类型，是红外还是可见光。
        else:
            return int(1)
    elif flag == 6: # SYSU
        if filename[0] == 'T':  # Thermal : 0
            return int(0)
        else:
            return int(1)