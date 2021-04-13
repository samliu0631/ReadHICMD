import yaml
from SimpleHICMD import SimpleHICMD
import argparse

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def set_domain_dis_param(opt):
    hyperparameters     = get_config('market2duke.yaml')    
    opt.id_dis_beta1    = hyperparameters['beta1']
    opt.id_dis_beta2    = hyperparameters['beta2']
    opt.id_dis_lr_id_d  = hyperparameters['lr_id_d']
    return opt
    
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
hyperparameters     = get_config('market2duke.yaml') 
opt = set_domain_dis_param(opt)
trainer = SimpleHICMD(opt,hyperparameters)



# if __name__=='__main__':
#     main()