import yaml
from  addnetwork import  IdDis


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def main():
    print('Hello world')
    hyperparameters = get_config('market2duke.yaml')
    id_dis = IdDis(hyperparameters['gen']['id_dim'], hyperparameters['dis'], fp16=False)


if __name__=='__main__':
    main()