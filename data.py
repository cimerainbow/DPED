import glob
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
# import h5py
import random

def dataset(cfgs, flag, trans):
    if cfgs['dataset'] == 'BSDS':
        dataset = BSDS_500(root=cfgs['dataset_path'], flag=flag, VOC=False, transform=trans)
    elif cfgs['dataset'] == 'BSDS-VOC':
        dataset = BSDS_500(root=cfgs['dataset_path'], flag=flag, VOC=True, transform=trans)
    elif cfgs['dataset'] == 'NYUD-image':
        dataset = NYUD(root=cfgs['dataset_path'], flag=flag, rgb=True, transform=trans)
    elif cfgs['dataset'] == 'NYUD-hha':
        dataset = NYUD(root=cfgs['dataset_path'], flag=flag, rgb=False, transform=trans)
    elif cfgs['dataset'] == 'MultiCue-Edge':
        dataset = MultiCue(root=cfgs['dataset_path'], flag=flag, edge=True, transform=trans, seq=cfgs['multicue_seq'])
    elif cfgs['dataset'] == 'MultiCue-Contour':
        dataset = MultiCue(root=cfgs['dataset_path'], flag=flag, edge=False, transform=trans, seq=cfgs['multicue_seq'])

    elif cfgs['dataset'] == 'PASCAL-VOC-12':
        dataset = PASCAL_VOC12(root=cfgs['dataset_path'], flag=flag, transform=trans)
    elif cfgs['dataset'] == 'PASCAL-Context':
        dataset = PASCAL_Context(root=cfgs['dataset_path'], flag=flag, transform=trans)
    else:
        raise NameError

    return dataset


'''  
#######################################################################################
BSDS_500
#######################################################################################
'''


def read_from_pair_txt(path, filename):
    path = path.replace('/', os.sep)
    pathfile = open(os.path.join(path, filename))  # 打开这个这个文件
    filenames = pathfile.readlines()
    pathfile.close()
    filenames = [f.strip() for f in filenames]  # 去除每行首尾空格，构成一个表格
    filenames = [c.split(' ') for c in filenames]  # 按中间空格把一个元素分成两个元素数组
    filenames = [(os.path.join(path, c[0].replace('/', os.sep)),  # 更改为esp兼容linux文件系统
                  os.path.join(path, c[1].replace('/', os.sep))) for c in filenames]  # 给每个相对路径变为绝对路径，
    return filenames


class BSDS_500(Dataset):
    def __init__(self, root, flag='train', VOC=False, transform=None):
        if flag == 'train':
            if VOC:
                filenames = read_from_pair_txt(root['BSDS-VOC'], 'bsds_pascal_train_pair_s5.lst')
            else:
                filenames = read_from_pair_txt(root['BSDS'], 'train_pair.lst')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1] for im_name in filenames]

        elif flag == 'test':
            self.im_list = glob.glob(os.path.join(root['BSDS'], r'test/*.jpg'))
            self.gt_list = [path.split(os.sep)[-1][:-4] for path in self.im_list]

        self.length = self.im_list.__len__()

        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            label = np.array(Image.open(self.gt_list[item]).convert('L'))
            label = Image.fromarray(label.astype(np.float32) / 255.0)
        elif self.flag == 'test':
            label = Image.open(self.im_list[item])

        sample = {'images': image, 'labels': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


'''  
#######################################################################################
NYUD-v2
#######################################################################################
'''


class NYUD(Dataset):
    def __init__(self, root, flag='train', rgb=True, transform=None):
        if flag == 'train':
            if rgb:
                filenames = read_from_pair_txt(root['NYUD-V2'], 'image-train.lst')
            else:
                filenames = read_from_pair_txt(root['NYUD-V2'], 'hha-train.lst')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1] for im_name in filenames]

        elif flag == 'test':
            if rgb:
                self.im_list = glob.glob(os.path.join(root['NYUD-V2'], 'test/Images/*.png'))
            else:
                self.im_list = glob.glob(os.path.join(root['NYUD-V2'], 'test/HHA/*.png'))
            self.gt_list = [path.split('/')[-1][:-4] for path in self.im_list]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            label = np.array(Image.open(self.gt_list[item]).convert('L'))
            label = Image.fromarray(label.astype(np.float32) / 255.0)
        elif self.flag == 'test':
            label = Image.open(self.im_list[item])

        sample = {'images': image, 'labels': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


'''  
#######################################################################################
MultiCue
#######################################################################################
'''


class MultiCue(Dataset):
    def __init__(self, root, flag='train', edge=True, transform=None, seq=1):
        filenames = read_from_pair_txt(root['multicue'], 'seq' + str(seq) + '.txt')
        self.im_list = [im_name[0] for im_name in filenames]
        self.gt_list = [im_name[1] for im_name in filenames]

        if flag == 'train':
            self.im_list = self.im_list[:80]
            self.gt_list = self.gt_list[:80]
        elif flag == 'test':
            self.im_list = self.im_list[80:]
            self.gt_list = [gt.split('/')[-1].split('.')[0] for gt in self.gt_list[80:]]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag
        self.edge = 'edges' if edge == True else 'boundaries'

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            h5 = h5py.File(self.gt_list[item], 'r')
            label = np.array(h5[self.edge]).astype(np.float32)
            label = Image.fromarray(np.mean(label, axis=0))
            h5.close()
        elif self.flag == 'test':
            label = Image.open(self.im_list[item])

        sample = {'images': image, 'labels': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


def make_random_multicue(MultiCue_path):
    lists = glob.glob(os.path.join(MultiCue_path, 'ground-truth/hdf5', '*.h5'))
    lists = [pth.replace(MultiCue_path, '') for pth in lists]
    for i in range(10):
        random.shuffle(lists)
        dset = [path.replace('ground-truth', 'images').replace('h5', 'png').replace('hdf5/', '') + ' ' + path
                for path in lists]

        with open(os.path.join(MultiCue_path, 'seq' + str(i + 1) + '.txt'), 'a') as file_handle:
            file_handle.write('\n'.join(dset))


'''  
#######################################################################################
PASCAL_VOC12
#######################################################################################
'''


class PASCAL_VOC12(Dataset):
    def __init__(self, root, flag='train', transform=None):
        if flag == 'train':
            filenames = self.read_file(root['PASCAL-VOC12'], 'train.txt')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1] for im_name in filenames]

        elif flag == 'test':
            filenames = self.read_file(root['PASCAL-VOC12'], 'val.txt')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1].split('.')[0].split('/')[-1] for im_name in filenames]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def read_file(self, path, filename):
        with open(os.path.join(path, 'ImageSets', 'Segmentation', filename)) as pfile:
            filenames = pfile.readlines()
        img_root = os.path.join(path, 'JPEGImages')
        gt_root = os.path.join(path, 'boundaries')
        filenames = [(os.path.join(img_root, f.strip() + '.jpg'), os.path.join(gt_root, f.strip() + '.png')) for f in
                     filenames]
        return filenames

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            label = np.array(Image.open(self.gt_list[item]).convert('L'))
            label = Image.fromarray(label.astype(np.float32) / 255.0)
        elif self.flag == 'test':
            label = Image.open(self.im_list[item])

        sample = {'images': image, 'labels': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


'''  
#######################################################################################
PASCAL_Context
#######################################################################################
'''


class PASCAL_Context(Dataset):
    def __init__(self, root, flag='train', transform=None):
        if flag == 'train':
            filenames = self.read_file(root['PASCAL-VOC12'], 'train_new.txt')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1] for im_name in filenames]

        elif flag == 'test':
            filenames = self.read_file(root['PASCAL-VOC12'], 'test_new.txt')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1].split('.')[0].split('/')[-1] for im_name in filenames]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def read_file(self, path, filename):
        with open(os.path.join(path, 'ImageSets', 'Context', filename)) as pfile:
            filenames = pfile.readlines()
        img_root = os.path.join(path, 'JPEGImages')
        gt_root = os.path.join(path, 'Context_label')
        filenames = [(os.path.join(img_root, f.strip() + '.jpg'), os.path.join(gt_root, f.strip() + '.png')) for f in
                     filenames]
        return filenames

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            label = np.array(Image.open(self.gt_list[item]).convert('L'))
            label = Image.fromarray(label.astype(np.float32) / 255.0)
        elif self.flag == 'test':
            label = Image.open(self.im_list[item])

        sample = {'images': image, 'labels': label}
        if self.transform:
            sample = self.transform(sample)
        return sample