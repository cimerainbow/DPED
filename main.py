import train
import transforms
import yaml
import dict_lossfuntion
import platform
import model_repo
import data
from torch import optim
import random
import numpy as np
import torch
import test
import os

def open_cfgs():
    if platform.system() == 'Windows':
        file_id = open('./Win_cfgs.yaml', encoding='utf-8')
    elif platform.system() == 'Linux':
        file_id = open('./Lin_cfgs.yaml', encoding='utf-8')

    cfgs = yaml.load(file_id, Loader=yaml.FullLoader)
    file_id.close()
    return cfgs

def seed_everthing(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    print('Seed everthing {}'.format(seed))

def config_optimizer(hyper,net,lr):
    if hyper['method'] == "SGD":
        optimizer = optim.SGD([{'params': net.parameters()}],
                              lr=lr, momentum=hyper['momentum'],
                              weight_decay=hyper['weight_decay']
                              )
    elif hyper['method'] == "Adam":
        optimizer = torch.optim.Adam([{'params': net.parameters()}],
                                     weight_decay=hyper['weight_decay']
                                     )
    else:
        raise NameError
    return optimizer

def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_steps=10):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (decay_rate ** (epoch // decay_steps))

def init_config(net, n_loss, n_lr, parameters=[0, 0],hyper=None):
    cfgs = open_cfgs()
    trans = transforms.Compose([
        # transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    net = net.Net(pretrain=True)
    if 'ns' in n_loss[0]:
        net.derepresentation()
        hyper['represent'] = 'ns'

    criterion = dict_lossfuntion.Loss(n_loss, parameters)
    optimizer = config_optimizer(hyper, net, n_lr)

    dataset = data.dataset(cfgs, "train", trans)
    # todo
    path_str = './' + net.name+'_'+criterion.name+'_'+hyper['method']+'_'+str(n_lr)
    config = {
        'cfgs': hyper,
        'net': net,
        'parameters': parameters,
        'criterion': criterion,
        'optimizer': optimizer,
        'dataset': dataset,
        'learning_rate_decay': learning_rate_decay,
        'path_str': path_str
    }
    return config

def test_init_config(net, pth_repo, test_path):
    cfgs = open_cfgs()
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    net = net.Net(pretrain=True).eval()
    dataset = data.dataset(cfgs, "test", trans)
    config = {
        'cfgs': cfgs,
        'net': net,
        'dataset': dataset,
        'pth_repo': pth_repo,
        'test_path': test_path,
        'pyramid': [0.5, 2]
    }
    return config 

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('当前有{}块GPU'.format(torch.cuda.device_count()))
    print('当前使用的GPU为{}'.format(torch.cuda.current_device()))

    HYPER = dict(
        batch_size=1,
        max_iter=4,
        decay_rate=0.1,
        decay_steps=2,
        # optimization Adam or SGD
        method='SGD',
        momentum=0.90,
        weight_decay=2.0e-4,
        represent='sigmoid'
    )

    nets =[
        model_repo.S83TOA_LJ,
            ]
    parameterses = (
        [1],
    )
    loss = [
        ['CEO_17_50'],
    ]
    lr =[
        1.0e-6,
    ]

    benchmark = True
    for n in nets:
        for i, (n_loss, n_lr) in enumerate(zip(loss, lr)):
            for parameters in parameterses:
                seed_everthing(78)
                config = init_config(n, n_loss, n_lr, parameters,hyper=HYPER)
                trainer = train.Train(config, benchmark)
                trainer.start()

                pth_repo = config['pth_repo']
                test_path = config['test_path']
                del trainer
                del config
                print('开始测试')
                test_configs = test_init_config(n, pth_repo, test_path)
                tester = test.Test(test_configs)
                tester.start()
                del test_configs
                del tester