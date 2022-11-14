import os, sys
import random
from PIL import Image
import numpy as np
import pandas as pd


class FaceDataset:
    # 初始化参数
    def __init__(self, csv_path, data_save_file='data.txt'):
        """
        :param csv_path: 图片路径
        :param data_save_file: 将图片转化为二维数据的文件名
        """
        self.csv_path = csv_path
        self.data_save_file = data_save_file

        self.w, self.h = 0, 0

        # save the data
        if not os.path.exists(self.data_save_file):
            self.preprocess_data()

    # 处理数据，将图片数据转化为二维矩阵
    def preprocess_data(self):
        try:
            with open(self.csv_path, 'r') as fp:
                lines = fp.readlines()
        except:
            raise Exception("Can not find the {}".format(self.csv_path))
        
        
        label_list, image_list = [], []
        for line in lines:
            image_path, label = line.split('\n')[0].split(';')
            label = np.array([int(label)]).reshape(1, -1)
            image = Image.open(image_path)
            image = np.reshape(np.asarray(image), (1, -1)) # [1, D]
            image_list.append(image)
            label_list.append(label)
            

        image_list = np.concatenate(image_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        data = np.column_stack((image_list, label_list))

        np.savetxt(self.data_save_file, data)


    # 加载读入数据
    def load_data(self):
        train_data = np.loadtxt(self.data_save_file)
        data = train_data[:, :-1]  # 取出特征数据
        target = train_data[:, -1]  # 取出标签数据
        return data, target


    def get_split_data(self, test_rate=0.3, shuffle=False):
        
        all_data = np.loadtxt(self.data_save_file)

        # 按照类别对数据进行划分
        target = all_data[:, -1]  # 取出标签数据
        n_classes = len(set(target)) # 40
        class_number = sum(all_data[:, -1] == 0)
        test_size = int(class_number * test_rate) 

        x_train, y_train, x_test, y_test = [], [], [], []
        for i in range(n_classes):
            lower = i * class_number
            upper = (i + 1) * class_number
            total_data = all_data[lower:upper, :]
            if shuffle:
                random.shuffle(total_data)

            x_tr, y_tr = total_data[:-test_size, :-1], total_data[:class_number-test_size, -1]
            x_te, y_te = total_data[class_number-test_size:, :-1], total_data[class_number-test_size:, -1]

            x_train.append(x_tr)
            y_train.append(y_tr)
            x_test.append(x_te)
            y_test.append(y_te)


        return np.concatenate(x_train, axis=0), np.concatenate(x_test, axis=0), np.concatenate(y_train, axis=0), np.concatenate(y_test, axis=0)
        # return np.concatenate([x_train], dim=0), np.array([x_test]), np.array([y_train]), np.array([y_test])



    def get_imgsize(self):
        try:
            with open(self.csv_path, 'r') as fp:
                lines = fp.readlines()
        except:
            raise Exception("Can not find the {}".format(self.csv_path))
        
        for line in lines:
            image_path, label = line.split('\n')[0].split(';')
            image = Image.open(image_path)
            self.w, self.h = image.size
            break
        return self.w, self.h
