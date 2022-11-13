import os, sys
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
