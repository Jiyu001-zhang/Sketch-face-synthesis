"""This module contains simple helper functions """
from __future__ import print_function

import time
from operator import itemgetter
import random
import torch
import numpy as np
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision
from models.networks import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (
        module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


# todo
def create_pixel_mapping(attaNum):
    img2_flat = attaNum.permute(0, 2, 3, 1).flatten(1, 2).cpu().detach().numpy()
    # nonzero_row_indices = (np.where(np.any(img2_flat[0] != 0, axis=1)))[0]
    # random_indices = np.random.choice(nonzero_row_index, size=min(256, len(nonzero_row_index)), replace=False)
    # np.sum(np.abs(img2_flat[0]),axis=1)
    # 建立像素之间的映射
    pixel_mapping = [(label, img2_val) for label, img2_val in
                     zip(list(range(0, img2_flat[0].shape[0])), list(np.sum(np.abs(img2_flat[0]), axis=1)))]

    my_array = np.array(pixel_mapping)
    sorted_indices = np.argsort(my_array[:, 1])[::-1]  # [::-1] 是对排序后的索引进行切片操作，使用切片操作[start:stop:step]
    sorted_array = my_array[sorted_indices]
    return sorted_array


def xyandpix(mapping_face, mapping_hair):
    start = time.time()
    ids = []
    # 将图像转换为NumPy数组
    scale_np_face = np.array(mapping_face)
    scale_np_hair = np.array(mapping_hair)

    # 获取图像的宽度和高度
    height_face, width_face = scale_np_face.shape
    height_hair, width_hair = scale_np_hair.shape

    # 创建一个二维数组来存储像素值，并生成坐标点网格
    face_x_coords, face_y_coords = np.meshgrid(np.arange(width_face), np.arange(height_face))
    hair_x_coords, hair_y_coords = np.meshgrid(np.arange(width_hair), np.arange(height_hair))
    face_pixel_values = scale_np_face[face_y_coords, face_x_coords]
    hair_pixel_values = scale_np_hair[hair_y_coords, hair_x_coords]

    # 将坐标点和像素值合并为一个字典
    face_pixels_dict = {(x, y): pixel_value for x, y, pixel_value in
                        zip(face_x_coords.flatten(), face_y_coords.flatten(), face_pixel_values.flatten())}

    # 将坐标点和像素值合并为一个字典
    hair_pixels_dict = {(x, y): pixel_value for x, y, pixel_value in
                        zip(hair_x_coords.flatten(), hair_y_coords.flatten(), hair_pixel_values.flatten())}
    # 取出噪点
    face_filtered_dict = {key: value for key, value in face_pixels_dict.items() if value > 0.4}  # 根据阈值过滤
    hair_filtered_dict = {key: value for key, value in hair_pixels_dict.items() if value > 0.4}  # 根据阈值过滤
    # 面部随机采点
    face_random_elements = random.sample(face_filtered_dict.items(), 256)
    # 将注意力值高的点多采一次
    double_face = [tup_face for tup_face in face_random_elements if tup_face[1] > 0.6]
    # 合并
    face = [tup_face_point[0] for tup_face_point in face_random_elements + double_face]

    hair_random_elements = random.sample(hair_filtered_dict.items(), 128)
    double_hair = [tup_hair for tup_hair in hair_random_elements if tup_hair[1] > 0.5]
    hair = [tup_hair_point[0] for tup_hair_point in hair_random_elements + double_hair]
    # 得到256*256图像中所有的点
    random_point_256 = face + hair

    # 等比代换
    id_256 = [key[0] * 256 + key[1] for key in random_point_256]
    id_262 = [(key[0] + 3) * 262 + (key[1] + 3) for key in random_point_256]
    id_128 = [int(key[0]/2) * 128 + int(key[1]/2) for key in random_point_256]
    id_64 = [int(key[0]/4) * 64 + int(key[1]/4) for key in random_point_256]
    ids.append(id_262)
    ids.append(id_256)
    ids.append(id_128)
    ids.append(id_64)
    # print(time.time()-start)
    return ids


from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn


class getFeaturemap():
    def __init__(self, src, mask, model, path='.'):
        # we will save the conv layer weights in this list
        self.model_weights = []
        # we will save the 49 conv layers in this list
        self.conv_layers = []
        self.model = model
        self.src = src
        self.mask = mask
        self.path = path
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(256),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,)),  # 归一化
                                             ])

    def __getconv2d(self):
        # get all the model children as list
        model_children = list(self.model.children())
        print(model_children)
        # counter to keep count of the conv layers
        counter = 0
        # append all the conv layers and their respective wights to the list
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                self.model_weights.append(model_children[i].weight)
                self.conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    # for child in model_children[i][j].children():
                    if type(model_children[i][j]) == nn.Conv2d:
                        counter += 1
                        self.model_weights.append(model_children[i][j].weight)
                        self.conv_layers.append(model_children[i][j])
                    if type(model_children[i][j]) == ResnetBlock:
                        model_children1 = list(model_children[i][j].children())
                        for k in range(len(model_children1[0])):
                            if type(model_children1[0][k]) == nn.Conv2d:
                                counter += 1
                                self.model_weights.append(model_children1[0][k].weight)
                                self.conv_layers.append(model_children1[0][k])

        print(f"Total convolution layers: {counter}")

    def imagePro(self):
        self.image = self.transform(self.src).unsqueeze(0).to(1)
        self.mask = self.transform(self.mask).unsqueeze(0).to(1)
        return self.image, self.mask

    def save_feature_map(self):
        self.__getconv2d()
        outputs = []
        names = []
        image, mask = self.imagePro()
        self.image = image * mask
        import matplotlib.pyplot as plt
        plt.imshow(self.image.squeeze(0).permute(1,2,0).cpu().detach().numpy())
        plt.savefig('0-b.png')
        for layer in self.conv_layers[0:]:
            self.image = layer(self.image)
            outputs.append(self.image)
            names.append(str(layer))

        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)  # 对通道数进行求和达到降维的作用
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())

        fig = plt.figure(figsize=(256, 256))
        for i in range(len(processed)):
            a = fig.add_subplot(24, 1, i + 1)
            imgplot = plt.imshow(processed[i])
            a.axis("off")
            a.set_title(names[i].split('(')[0], fontsize=30)
        plt.savefig(str(f'{self.path}/feature_mapsB.jpg'), bbox_inches='tight')
