import os
import torch
from pycocotools import coco
import queue
import threading
from model.model_image import build_model, weights_init
from tools import custom_print
from data_processed import train_data_producer
from train import train
import time
import argparse
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # train_val_config 训练集、验证集、超参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--vgg16_path', default='./weights/vgg16_bn_feat.pth',help="vgg path")
    parser.add_argument('--npy_path',default='./utils/new_cat2imgid_dict4000.npy', help="npy path")
    parser.add_argument('--output_dir', default='./VSOD_results/wo_optical_flow/DAVIS/', help='directory for result')
    parser.add_argument('--gpu_id', default='cuda:2', help='id of gpu')
    parser.add_argument('--img_size', default=224, help='image size')
    parser.add_argument('--lr', default=1e-5, help='learning rate')
    parser.add_argument('--lr_de', default=20000, help='learning rate decay')
    parser.add_argument('--epochs', default=100000, help='epochs')
    parser.add_argument('--bs', default=2, help='batch size')
    parser.add_argument('--gs', default=5, help='group size')
    parser.add_argument('--log_interval', default=100, help='log interval')
    parser.add_argument('--val_interval', default=1000, help='val interval')
    args = parser.parse_args()
    
    annotation_file = '/data1/datasets/coco2017/annotations/instances_train2017.json'
    coco_item = coco.COCO(annotation_file=annotation_file)

    train_datapath = '/data1/datasets/coco2017/train2017/'

    # val_datapath = ['/data1/datasets/UFO/VOCdevkit/VOC2007/JPEGImages/']
    val_datapath = ['/data1/datasets/UFO/datasets/iCoseg8',
                    '/data1/datasets/UFO/datasets/MSRC7',
                    '/data1/datasets/UFO/datasets/Internet_Datasets300',
                    '/data1/datasets/UFO/datasets/PASCAL_VOC']

    vgg16_path = args.vgg16_path
    npy = args.npy_path

    # project config
    project_name = 'UFO'
    device = torch.device('cuda:2')
    img_size = args.img_size
    lr = args.lr
    lr_de = args.lr_de
    epochs = args.epochs
    batch_size = args.bs
    group_size = args.gs
    log_interval = args.log_interval
    val_interval = args.val_interval

    # create log dir
    log_root = './logs'
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    # create log txt 训练记录保存
    log_txt_file = os.path.join(log_root, project_name + '_log.txt')
    custom_print(project_name, log_txt_file, 'w')

    # create model save dir 模型保存
    models_root = './models'
    if not os.path.exists(models_root):
        os.makedirs(models_root)

    models_train_last = os.path.join(models_root, project_name + '_last.pth')
    models_train_best = os.path.join(models_root, project_name + '_best.pth')

    net = build_model(device).to(device)
    net.train()
    net.apply(weights_init)
    net.base.load_state_dict(torch.load(vgg16_path))
    # print(net)

    # net = net.to(device)
    # net = torch.nn.DataParallel(net)

    # continute load checkpoint
    # net.load_state_dict(torch.load('./models/SSNM-Coseg_last.pth', map_location='cuda:0'))

    q = queue.Queue(maxsize=40)

    p1 = threading.Thread(target=train_data_producer, args=(coco_item, train_datapath, npy, q, batch_size, group_size, img_size))
    p2 = threading.Thread(target=train_data_producer, args=(coco_item, train_datapath, npy, q, batch_size, group_size, img_size))
    p3 = threading.Thread(target=train_data_producer, args=(coco_item, train_datapath, npy, q, batch_size, group_size, img_size))
    p1.start()
    p2.start()
    p3.start()
    time.sleep(2)

    train(net, device, q, log_txt_file, val_datapath, models_train_best, models_train_last, lr, lr_de, epochs, log_interval, val_interval)
