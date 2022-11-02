import torch
import time
import os
from datetime import datetime
import data
import cv2
from matplotlib import pyplot as plt
import numpy as np
# from tqdm import tqdm
from multiprocessing import cpu_count
import glob
# from apex import amp


def init_path(config):
    pth_repo = config['path_str'] + '/'
    parameters = config['parameters']
    pth_repo = pth_repo + 'param_{}/'.format(parameters[0])
    if not os.path.exists(pth_repo):
        os.makedirs(pth_repo)

    test_path = config['path_str'] + '/'
    test_path = test_path + 'param_{}/'.format(parameters[0])
    temp_list = test_path.split('/')
    test_path = temp_list[0] + '/' + temp_list[1] + '/' + temp_list[2] + r'_test/'
    validation_path = test_path + 'validation/'
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)
    config['pth_repo'] = pth_repo
    config['test_path'] = test_path
    return pth_repo, validation_path

class Train:
    def __init__(self, config, benchmark=False):
        self.info = config['cfgs']
        self.parameters = config['parameters']
        self.represet = self.info['represent']
        self.net = config['net']
        self.criterion = config['criterion']
        self.optimizer = config['optimizer']
        self.dataset = config['dataset']
        self.learning_rate_decay = config['learning_rate_decay']

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      shuffle=True,
                                                      batch_size=self.info['batch_size'],
                                                      num_workers=4)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # todo 定义保存的文件路径
        sava_path, validation_path = init_path(config)
        self.sava_path = sava_path
        self.validation_path = validation_path
        self.print_staistaic_text = validation_path + 'print_staistaic_text.txt'
        self.benchmark = benchmark
    def start(self):
        if self.benchmark:
            print_setp, validation_step = [2, 4]
        else:
            print_setp, validation_step = [200, 400]
        self.net.to(self.device)

        # self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O1', loss_scale='128.0')

        self.criterion.to(self.device)
        epochs = self.info['max_iter']
        # 断点续传
        start_epoch = self.checkpoint()

        for epoch in range(epochs)[start_epoch:]:
            self.learning_rate_decay(self.optimizer, epoch, decay_rate=self.info['decay_rate'], decay_steps=self.info['decay_steps'])
            running_loss = 0.0
            f1_rate = 0.0
            f2_rate = 0.0
            loss_rate_all = 0
            # pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
            for i, data in enumerate(self.dataloader):
                # todo
                # if i == 10:
                #     break  #   测试code
                # todo

                start_time = time.time()
                self.optimizer.zero_grad()
                images = data['images'].to(self.device)
                labels = data['labels'].to(self.device)

                prediction = self.net(images)
                loss, func_rate, loss_rate = self.criterion(prediction, labels)
                loss.backward()
                self.optimizer.step()
                # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                #     scaled_loss.backward()
                duration = time.time() - start_time
                running_loss += loss.item()

                if func_rate.shape[0] == 1:
                    f1_rate += func_rate[0].item()
                    loss_rate_all += loss_rate.item()
                else:
                    f1_rate += func_rate[0].item()
                    f2_rate += func_rate[1].item()
                    loss_rate_all += loss_rate.item()

                # print_setp = 30
                if i % print_setp == print_setp - 1 or i == len(self.dataloader):
                    # s = ('epoch:%g/%g loss = %.3f ') % (epoch, epochs - 1, running_loss)
                    # pbar.set_description(s)
                    rat_str = 'f1_rate:%.2f f2_rate:%.2f loss_rate:%.2f'
                    print(rat_str % (f1_rate / print_setp,
                                     f2_rate / print_setp,
                                     loss_rate_all/ print_setp)
                          )

                    examples_per_sec = self.info['batch_size'] / duration
                    sec_per_batch = float(duration)
                    format_str = "%s: step [%d, %4d/%4d], loss = %.3f lr =%f  (%.1f examples/sec; %.3f sec/batch)"
                    print(format_str % (datetime.now(),
                                        epoch + 1, i + 1, len(self.dataloader),
                                        running_loss / print_setp,
                                        self.optimizer.state_dict()['param_groups'][0]['lr'],
                                        examples_per_sec, sec_per_batch))

                    # save print_staistaic
                    file_handle = open(self.print_staistaic_text, mode='a')

                    file_handle.write(rat_str % (f1_rate / print_setp,
                                     f2_rate / print_setp,
                                     (f1_rate / print_setp) / (f1_rate / print_setp + f2_rate / print_setp)))
                    file_handle.write('\n')
                    file_handle.write(format_str % (datetime.now(),
                                                  epoch + 1, i + 1, len(self.dataloader),
                                                  running_loss / print_setp,
                                                  self.optimizer.state_dict()['param_groups'][0]['lr'],
                                                  examples_per_sec, sec_per_batch))
                    file_handle.write('\n')
                    file_handle.close()
                    running_loss = 0.0
                    f1_rate = 0.0
                    f2_rate = 0.0

                # validation_step = 50
                if i % validation_step == validation_step - 1 or i == len(self.dataloader):
                    prediction = self.net(images)
                    if self.represet == 'ns':
                        if isinstance(prediction, (list, tuple)):
                            prediction = prediction[0].sigmoid().cpu().detach().numpy().transpose((0, 2, 3, 1))
                        else:
                            prediction = prediction.sigmoid().cpu().detach().numpy().transpose((0, 2, 3, 1))
                    else:
                        if isinstance(prediction, (list, tuple)):
                            prediction = prediction[0].cpu().detach().numpy().transpose((0, 2, 3, 1))
                        else:
                            prediction = prediction.cpu().detach().numpy().transpose((0, 2, 3, 1))

                    pred_flat = prediction.flatten()
                    labels_flat = labels.cpu().detach().numpy().transpose((0, 2, 3, 1)).flatten()
                    data_pos = pred_flat[labels_flat > 0]
                    data_neg = pred_flat[labels_flat == 0]

                    for j in range(1):
                        cv2.imwrite(self.validation_path+'epoch'+str(epoch)+'_'+str(j)+'.png', prediction[j] * 255)

                        ax = plt.subplot(2, 2, 1)
                        p_count, bin, _ = ax.hist(data_pos, bins=np.linspace(0, 1, 100, endpoint=True))
                        ax = plt.subplot(2, 2, 2)
                        n_count, bin, _ = ax.hist(data_neg, bins=np.linspace(0, 1, 100, endpoint=True))

                        ax = plt.subplot(2, 2, 3)
                        ax.set_ylim(0, 0.04)
                        # p_count, bin = np.histogram(data_pos, 100, (0, 1))  # hist1 每个灰度值的频数
                        p_count = p_count / len(pred_flat)
                        ax.bar(bin[:-1], p_count, width=0.01)

                        ax = plt.subplot(2, 2, 4)
                        ax.set_ylim(0, 0.16)
                        # n_count, bin = np.histogram(data_neg, 100, (0, 1))  # hist1 每个灰度值的频数
                        n_count = n_count / len(pred_flat)
                        ax.bar(bin[:-1], n_count, width=0.01)
                        plt.savefig(self.validation_path + 'test' + str(epoch) + '.png')
                        plt.close('all')
             
            self.sava_model(epoch)        
        print("Finish Training")


    def checkpoint(self):
        pth_list = glob.glob(os.path.join(self.sava_path, r'*.pth'))

        if len(pth_list) == 0:
            print('没有断点，正常训练')
            start_epoch = 0
        else:
            epoch = pth_list[-1][-5]
            print(self.sava_path + 'epoch' + str(epoch) + '.pth')
            interrupt = torch.load(self.sava_path + 'epoch' + str(epoch) + '.pth')
            self.net.load_state_dict(interrupt['model'])  # 加载模型的可学习参数
            self.optimizer.load_state_dict(interrupt['optimizer'])  # 加载优化器参数
            start_epoch = interrupt['epoch'] + 1  # 设置开始的epoch
            del interrupt
            print('已恢复训练')
        return start_epoch

    def sava_model(self, epoch):
        # torch.save(self.net.state_dict(), self.sava_path + 'epoch'+str(epoch))
        state = {'model': self.net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'epoch': epoch,
                 'lr': self.optimizer.state_dict()['param_groups'][0]['lr']}  # epoch从0开始
        torch.save(state, self.sava_path + 'epoch' + str(epoch) + '.pth')

