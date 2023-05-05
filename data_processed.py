import torch
from torchvision import transforms
import copy
import random
import os
import numpy as np
from PIL import Image
import imgviz
from imageio import imread
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2


def save_colored_mask(mask, save_path='./maskVis'):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def filt_small_instance(coco_item, pixthreshold=4000, imgNthreshold=5):
    list_dict = coco_item.catToImgs
    for catid in list_dict:
        list_dict[catid] = list(set(list_dict[catid]))
    new_dict = copy.deepcopy(list_dict)
    for catid in list_dict:
        imgids = list_dict[catid]
        for n in range(len(imgids)):
            imgid = imgids[n]
            anns = coco_item.imgToAnns[imgid]
            has_large_instance = False
            for ann in anns:
                if (ann['category_id'] == catid) and (ann['iscrowd'] == 0) and (ann['area'] > pixthreshold):
                    has_large_instance = True
            if has_large_instance is False:
                new_dict[catid].remove(imgid)
        imgN = len(new_dict[catid])
        if imgN < imgNthreshold:
            new_dict.pop(catid)
            print('catid:%d  remain %d images, delet it!' % (catid, imgN))
        else:
            print('catid:%d  remain %d images' % (catid, imgN))
    print('remain  %d  categories' % len(new_dict))
    np.save('./utils/new_cat2imgid_dict%d.npy' % pixthreshold, new_dict)
    return new_dict


def train_data_producer(coco_item, datapath, npy, q, batch_size=10, group_size=5, img_size=224):
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gt_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.449], std=[0.226])])
    if os.path.exists(npy):
        # list_dict = np.load(npy).item()
        list_dict = np.load(npy, allow_pickle=True).item()
    else:
        list_dict = filt_small_instance(coco_item, pixthreshold=4000, imgNthreshold=100)
    catid2label = {}
    n = 0
    for catid in list_dict:
        catid2label[catid] = n
        n = n + 1
    # print(catid2label)
    # {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13,
    # 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    # 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38,
    # 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51,
    # 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64,
    # 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 81: 70, 82: 71, 84: 72, 85: 73, 86: 74, 87: 75, 88: 76, 90: 77}

    while 1:
        rgb = torch.zeros(batch_size * group_size, 3, img_size, img_size)
        cls_labels = torch.zeros(batch_size, 78)
        mask_labels = torch.zeros(batch_size * group_size, img_size, img_size)
        # print('len(list_dict)', len(list_dict))  # 78
        # print('list_dict.keys()', list_dict.keys())
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        # 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
        # 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
        # 78, 79, 81, 82, 84, 85, 86, 87, 88, 90]
        if batch_size > len(list_dict):
            remainN = batch_size - len(list_dict)
            batch_catid = random.sample(list_dict.keys(), remainN) + random.sample(list_dict, len(list_dict))
        else:
            batch_catid = random.sample(list_dict.keys(), batch_size)
        group_n = 0
        img_n = 0
        for catid in batch_catid:
            imgids = random.sample(list_dict[catid], group_size)
            co_catids = []
            anns = coco_item.imgToAnns[imgids[0]]
            # region
            # print('imgids[0]', imgids[0])
            # print('anns', anns)
            #  [{'segmentation': [[536.91, 268.35, 529.82, 249.59, 486.73, 251.11, 491.8, 256.18, 535.39, 269.36]],
            #  'area': 525.1052500000001, 'iscrowd': 0,
            #  'image_id': 278931, 'bbox': [486.73, 249.59, 50.18, 19.77],
            #  'category_id': 77, 'id': 324116},
            #  {'segmentation': [[382.33, 583.12, 368.58, 496.48, 408.46, 462.09, 404.33, 396.08, 374.08, 331.44, 358.95, 242.05, 328.69, 192.54, 284.68, 182.91, 244.8, 166.41, 225.55, 118.27, 189.79, 100.4, 140.28, 112.77, 129.28, 167.78, 140.28, 222.8, 118.27, 279.18, 122.4, 338.32, 115.52, 386.45, 71.51, 408.46, 49.51, 424.96, 46.76, 462.09, 52.26, 598.25, 127.9, 598.25, 127.9, 497.85, 130.65, 474.47, 166.41, 482.72, 195.29, 504.73, 290.18, 489.6, 306.69, 601.0, 385.08, 596.87]],
            #  'area': 109216.12415000002, 'iscrowd': 0,
            #  'image_id': 278931, 'bbox': [46.76, 100.4, 361.7, 500.6],
            #  'category_id': 1, 'id': 450632},
            #  {'segmentation': [[427.3, 583.05, 413.51, 516.89, 406.62, 483.81, 413.51, 447.97, 427.3, 419.03, 432.81, 390.08, 432.81, 339.08, 430.05, 311.51, 439.7, 271.54, 452.11, 245.35, 463.14, 230.19, 464.51, 208.14, 475.54, 194.35, 481.05, 162.65, 479.68, 137.84, 482.43, 110.27, 510.0, 78.57, 529.3, 60.65, 545.84, 59.27, 559.62, 60.65, 580.3, 62.03, 585.81, 66.16, 591.32, 591.32, 527.92, 598.22, 428.68, 600.97]],
            #  'area': 77061.854, 'iscrowd': 0,
            #  'image_id': 278931, 'bbox': [406.62, 59.27, 184.7, 541.7],
            #  'category_id': 1, 'id': 514928},
            #  {'segmentation': [[38.49, 393.55, 30.44, 403.84, 58.18, 415.47, 70.71, 425.76, 104.71, 444.55, 114.11, 449.47, 115.9, 447.68, 120.37, 450.82, 146.77, 439.18, 155.27, 432.02, 152.59, 427.1, 116.79, 402.05, 110.53, 398.02, 96.66, 405.63, 90.84, 409.21, 76.08, 410.1, 49.23, 401.6, 39.84, 392.65], [192.41, 467.82, 193.3, 469.61, 186.15, 479.0, 179.88, 485.27, 165.56, 485.72, 160.64, 480.35, 174.51, 477.66, 183.91, 473.64]],
            #  'area': 3248.3276499999997, 'iscrowd': 0,
            #  'image_id': 278931, 'bbox': [30.44, 392.65, 162.86, 93.07],
            #  'category_id': 84, 'id': 1138315},
            #  {'segmentation': [[445.75, 411.45, 426.15, 425.02, 414.08, 434.07, 406.55, 442.36, 414.84, 454.42, 405.04, 464.22, 395.24, 473.27, 386.94, 492.87, 396.74, 519.26, 406.55, 529.81, 585.97, 529.06, 590.49, 521.52, 592.76, 513.98, 594.26, 422.0, 593.51, 402.4, 587.48, 393.36, 563.35, 382.8, 528.68, 382.05, 511.34, 382.8, 463.84, 393.36, 444.99, 408.43]],
            #  'area': 25637.02594999999, 'iscrowd': 0,
            #  'image_id': 278931, 'bbox': [386.94, 382.05, 207.32, 147.76],
            #  'category_id': 31, 'id': 1836207},
            #  {'segmentation': [[120.67, 372.1, 123.31, 378.72, 120.0, 386.66, 102.79, 400.57, 94.85, 404.54, 90.21, 403.21, 79.62, 408.51, 53.8, 404.54, 39.9, 393.28, 29.97, 403.88, 41.22, 406.52, 61.08, 416.45, 49.83, 423.74, 47.84, 457.5, 41.22, 456.84, 41.89, 436.31, 43.21, 430.36, 35.27, 417.78, 25.34, 412.48, 24.01, 266.18, 26.0, 250.29, 47.84, 252.27, 73.66, 266.18, 92.86, 299.28, 104.11, 337.67, 104.78, 346.28, 104.78, 345.62, 108.75, 360.0, 118.02, 364.63]],
            #  'area': 11714.6193, 'iscrowd': 0,
            #  'image_id': 278931, 'bbox': [24.01, 250.29, 99.3, 207.21],
            #  'category_id': 1, 'id': 2198513}]
            # endregion
            for ann in anns:
                if (ann['iscrowd'] == 0) and (ann['area'] > 4000):
                    co_catids.append(ann['category_id'])
            co_catids_backup = copy.deepcopy(co_catids)
            # print('co_catids', co_catids)
            for imgid in imgids[1:]:
                img_catids = []
                anns = coco_item.imgToAnns[imgid]
                for ann in anns:
                    if (ann['iscrowd'] == 0) and (ann['area'] > 4000):
                        img_catids.append(ann['category_id'])
                for co_catid in co_catids_backup:
                    if co_catid not in img_catids:
                        co_catids.remove(co_catid)
                co_catids_backup = copy.deepcopy(co_catids)
            # print('co_catids', co_catids)
            for co_catid in co_catids:
                cls_labels[group_n, catid2label[co_catid]] = 1

            # print('cls_labels', cls_labels.size())
            # print('imgids', imgids)
            for imgid in imgids:
                # print('imgid', imgid)
                path = datapath + '%012d.jpg' % imgid
                # print('path', path)
                img = Image.open(path)
                if img.mode == 'RGB':
                    img = img_transform(img)
                else:
                    img = img_transform_gray(img)
                anns = coco_item.imgToAnns[imgid]
                mask = None
                for ann in anns:
                    if ann['category_id'] in co_catids:
                        if mask is None:
                            mask = coco_item.annToMask(ann)
                        else:
                            mask = mask + coco_item.annToMask(ann)
                mask[mask > 0] = 255
                mask = Image.fromarray(mask)
                mask = gt_transform(mask)
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0

                rgb[img_n, :, :, :] = copy.deepcopy(img)
                mask_labels[img_n, :, :] = copy.deepcopy(mask)
                img_n = img_n + 1

            group_n = group_n + 1

        idx = mask_labels[:, :, :] > 1
        mask_labels[idx] = 1
        # print('rgb.size====================', rgb.size())  # [40,3,224,224]
        # print('cls_labels.size====================', cls_labels.size())  # [8,78]
        # print('cls_labels====================', cls_labels)
        # print('mask_labels.size====================', mask_labels.size())  # [40,3,224,224]
        q.put([rgb, cls_labels, mask_labels])


def img_normalize(image):
    if len(image.shape) == 2:
        channel = (image[:, :, np.newaxis] - 0.485) / 0.229
        image = np.concatenate([channel, channel, channel], axis=2)
    else:
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))) \
                / np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return image


davis_fbms = ['bear', 'bear01', 'bear02', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'cars2', 'cars3',
              'cars6', 'cars7', 'cars8', 'cars9', 'cats02', 'cats04', 'cats05', 'cats07', 'dance-jump', 'dog-agility',
              'drift-turn', 'ducks01', 'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low', 'horses01',
              'horses03', 'horses06', 'kite-walk', 'lion02', 'lucia', 'mallard-fly', 'mallard-water', 'marple1',
              'marple10', 'marple11', 'marple13', 'marple3', 'marple5', 'marple8', 'meerkats01', 'motocross-bumps',
              'motorbike', 'paragliding', 'people04', 'people05', 'rabbits01', 'rabbits05', 'rhino', 'rollerblade',
              'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']


class VideoDataset(Dataset):
    def __init__(self, dir_, epochs, size=224, group=5, use_flow=False):

        self.img_list = []
        self.label_list = []
        self.flow_list = []

        self.group = group

        dir_img = os.path.join(dir_, 'image')
        dir_gt = os.path.join(dir_, 'groundtruth')
        dir_flow = os.path.join(dir_, 'flow')
        self.dir_list = sorted(os.listdir(dir_img))
        self.leng = 0
        for i in range(len(self.dir_list)):
            ok = 0
            if self.dir_list[i] in davis_fbms:
                ok = 1
            if ok == 0:
                continue
            tmp_list = []
            cur_dir = sorted(os.listdir(os.path.join(dir_img, self.dir_list[i])))
            for j in range(len(cur_dir)):
                tmp_list.append(os.path.join(dir_img, self.dir_list[i], cur_dir[j]))
            self.leng += len(tmp_list)
            self.img_list.append(tmp_list)

            tmp_list = []
            cur_dir = sorted(os.listdir(os.path.join(dir_gt, self.dir_list[i])))
            for j in range(len(cur_dir)):
                tmp_list.append(os.path.join(dir_gt, self.dir_list[i], cur_dir[j]))
            self.label_list.append(tmp_list)

        self.img_size = 224
        self.dataset_len = epochs
        self.use_flow = use_flow
        self.dir_ = dir_

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        rd = np.random.randint(0, len(self.img_list))
        rd2 = np.random.permutation(len(self.img_list[rd]))
        cur_img = []
        cur_flow = []
        cur_gt = []
        for i in range(self.group):
            cur_img.append(self.img_list[rd][rd2[i % len(self.img_list[rd])]])
            cur_flow.append(
                os.path.join(self.dir_, 'flow', os.path.split(self.img_list[rd][rd2[i % len(self.img_list[rd])]])[1]))
            cur_gt.append(self.label_list[rd][rd2[i % len(self.img_list[rd])]])

        group_img = []
        group_flow = []
        group_gt = []
        for i in range(self.group):
            tmp_img = imread(cur_img[i])

            tmp_img = torch.from_numpy(img_normalize(tmp_img.astype(np.float32) / 255.0))
            tmp_img = F.interpolate(tmp_img.unsqueeze(0).permute(0, 3, 1, 2), size=(self.img_size, self.img_size))
            group_img.append(tmp_img)

            tmp_gt = np.array(Image.open(cur_gt[i]).convert('L'))
            tmp_gt = torch.from_numpy(tmp_gt.astype(np.float32) / 255.0)
            tmp_gt = F.interpolate(tmp_gt.view(1, tmp_gt.shape[0], tmp_gt.shape[1], 1).permute(0, 3, 1, 2),
                                   size=(self.img_size, self.img_size)).squeeze()
            tmp_gt = tmp_gt.view(1, tmp_gt.shape[0], tmp_gt.shape[1])
            group_gt.append(tmp_gt)
            if self.use_flow == True:
                tmp_flow = imread(cur_flow[i])
                tmp_flow = torch.from_numpy(img_normalize(tmp_flow.astype(np.float32) / 255.0))
                tmp_flow = F.interpolate(tmp_flow.unsqueeze(0).permute(0, 3, 1, 2), size=(self.img_size, self.img_size))
                group_flow.append(tmp_flow)

        group_img = (torch.cat(group_img, 0))
        if self.use_flow == True:
            group_flow = torch.cat(group_flow, 0)
        group_gt = (torch.cat(group_gt, 0))
        if self.use_flow == True:
            return group_img, group_flow, group_gt
        else:
            return group_img, group_gt
