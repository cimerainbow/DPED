import data
import torch
import torch.nn.functional as F
import cv2
import os
import time
import numpy as np
import glob

def handle_path(pth_repo):
    pth_list = glob.glob(os.path.join(pth_repo, r'*.pth').replace('/', os.sep))
    return pth_list


class Test:
    def __init__(self, configs):
        self.t_time = 0.0
        self.t_sec = 0.0
        self.net = configs['net']
        self.info = configs['cfgs']
        self.dataset = configs['dataset']

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pth_repo = configs['pth_repo']
        self.test_path = configs['test_path']
        self.pyramid = configs['pyramid']

    def start(self):
        print(self.pth_repo)
        pth_list = handle_path(self.pth_repo)
        print(pth_list)
        pth_list = [pth_list[-1]]
        print('Finding......./n')
        print(pth_list)
        for pth in pth_list:
            print("Loading .......   path:{}".format(pth))
            self.net.load_state_dict(torch.load(pth)['model'])
            temp_list = pth.split(os.sep)
            save_path = self.test_path + temp_list[-1][:-4] + '/'
      
            self.net.to(self.device)
            for i, data in enumerate(self.dataloader, 0):
                with torch.no_grad():
                    images = data['images'].to(self.device)
                    w, h = data['images'].size()[2:]
                    factor = self.pyramid
                    images_15 = F.interpolate(data['images'], scale_factor=factor[1], mode='bilinear', align_corners=False)
                    images_half = F.interpolate(data['images'], scale_factor=factor[0], mode='bilinear', align_corners=False)
                    star_time = time.time()

                    images = images.to(self.device)
                    prediction = self.forward(images)

                    images_15 = images_15.to(self.device)
                    prediction_15 = self.forward(images_15)
                    prediction_15X = self.resize(prediction_15, w, h)

                    images_half = images_half.to(self.device)
                    prediction_half = self.forward(images_half)
                    prediction_halfX = self.resize(prediction_half, w, h)
                    output = (prediction + prediction_15X + prediction_halfX) / 3

                    duration = time.time() - star_time
                    self.t_time += duration
                    self.t_sec += 1 / duration
                    print('Process %3d/%3d image.' % (i+1, self.dataset.length))
                    
                    if not os.path.exists(save_path + '1X/'):
                        print(save_path)
                        os.makedirs(save_path + '1X/')
                    if not os.path.exists(save_path + 'multi/'):
                        os.makedirs(save_path + 'multi/')
                    
                    cv2.imwrite(save_path+'1X/' + self.dataset.gt_list[i]+'.png', prediction*255)
                    cv2.imwrite(save_path+'multi/' + self.dataset.gt_list[i] + '.png', output * 255)

            print('avg_time:%.3f, avg_FPS:%.3f' % (self.t_time / self.dataset.length, self.t_sec / self.dataset.length ))
    
    def forward(self, images):
        prediction = self.net(images)
        if isinstance(prediction, (list, tuple)):
            result = prediction[0]
        else:
            result = prediction
        prediction = result.cpu().detach().numpy().squeeze()
        return prediction

    def resize(self, image, w, h):
        if isinstance(image, (list, tuple)):
            for i in range(len(image)):
                image[i] = cv2.resize(image[i], (h, w), interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, (h, w), interpolation=cv2.INTER_CUBIC)

        return image

    def fuzze(self, S, S_15, S_half):
        if isinstance(S, (list,tuple)):
            fuzze = []
            for i in range(len(S)):
                fuzze.append((S[i]+S_15[i]+S_half[i])/3)
        else:
            fuzze = (S + S_15 + S_half) / 3
        return fuzze