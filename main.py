import os, argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from dataset import FaceDataset
from model import PCAModel,SVMModel, KNNModel

from visualization import plot_curve, get_title, plot_gallery
import utils

import warnings
warnings.filterwarnings('ignore')


def get_args():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default='./configs/run_video_retrieval.yaml')
    parser.add_argument('--csv_path', default='/home/wangcong/Course/PRML/PCA_PRML/data/orl_faces.csv')
    parser.add_argument('--data_save_file', default='/home/wangcong/Course/PRML/PCA_PRML/data/data.txt')
    parser.add_argument('--output_path', default='/home/wangcong/Course/PRML/PCA_PRML/output')

    # addition setting
    parser.add_argument('--random_state', default=5, type=int, help='Random number')
    parser.add_argument('--test_size', default=0.3, type=float, help='The rate of test data')

    # PCA setting
    parser.add_argument('--eigenvalue_path', default='/home/wangcong/Course/PRML/PCA_PRML/data/Eigenvalues.txt')
    parser.add_argument('--pca_ncomponents', default=399, type=float, help='The rate of test data, -1 do not use pca before svm')


    # SVM setting
    parser.add_argument('--svm_ncomponents', default=20, type=float, help='The rate of test data, -1 do not use pca before svm')

    # KNN setting
    parser.add_argument('--knn_ncomponents', default=50, type=float, help='The rate of test data, -1 do not use pca before svm')


    args = parser.parse_args()

    return args



if __name__ == '__main__':

   args = get_args()
   utils.set_seed(args.random_state)
   
   
   faceDataset = FaceDataset(args.csv_path, data_save_file=args.data_save_file)
   x_data, y_data = faceDataset.load_data()
   w, h = faceDataset.get_imgsize()
   x_train, x_test, y_train, y_test = faceDataset.get_split_data(test_rate=0.2, shuffle=True)
   x_test_plot = x_test.reshape(-1, h, w)

   # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=args.test_size, random_state=args.random_state)

   # print(x_train.shape)
   # print(x_test.shape)
   # print(y_train.shape)
   # print(y_test.shape)
   
   n_classes = len(set(sorted(y_test)))
   target_names = ["person" + str(i) for i in range(1, 1+n_classes)]


   # print(set(sorted(y_test)), len(set(sorted(y_test))))

   ################################################## PCA ##################################################

#  eigenvalues = utils.read_txtdata(args.eigenvalue_path)
#  eigenvalues = utils.normal_data(eigenvalues)

#  pca_output = os.path.join(args.output_path, 'PCA')
#  utils.exist_make(pca_output)
   
#  threshold = [0.9, 0.95, 0.99]
#  values, idxs = [], []
#  for thres in threshold:
#     value, idx = eigenvalues[np.where(eigenvalues > thres)].min(), np.argwhere(eigenvalues > thres).min()
#     values.append(value)
#     idxs.append(idx)

#  plot_curve(eigenvalues, values=values, idxs=idxs, output=os.path.join(pca_output, 'eigenvalue.jpg'))

#  # model
#  pca = PCAModel(x_data, y_data, None, None, args.pca_ncomponents)
#  explained_variance_ratio_ = pca.train_model()
#  explained_variance_ratio_ = utils.normal_data(explained_variance_ratio_)

#  threshold = [0.9, 0.95, 0.99]
#  values, idxs = [], []
#  for thres in threshold:
#     value, idx = eigenvalues[np.where(explained_variance_ratio_ > thres)].min(), np.argwhere(explained_variance_ratio_ > thres).min()
#     values.append(value)
#     idxs.append(idx)

#  plot_curve(explained_variance_ratio_, values=values, idxs=idxs, output=os.path.join(pca_output, 'pca_eigenvalue.jpg'))

   # print(y_data)
   # print(list(set(y_data)))
   # y_data = label_binarize(y_data, classes=list(set(y_data)))
   # print(y_data)

   # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=args.test_size, random_state=args.random_state)

   ################################################## SVM ##################################################
   
   svm_output_path = './output/SVM1'
   svm = SVMModel(x_train, x_test, y_train, y_test, output_path=svm_output_path)
   model, pca = svm.PCASearch(ncomponents=list(range(5, 61)))

   print("Best Parameter by search")
   print("pca: ", pca)
   print(model)

   
   score, y_predict, y_proba = svm.test_model(model, pca)
   print('Acc: {}'.format(score))
   svm.evaluate(target_names)

   prediction_titles = [get_title(y_predict, y_test, target_names, i)
                  for i in range(y_predict.shape[0])]
   
   plot_gallery(x_test_plot, prediction_titles, h, w, save_path=svm_output_path)


   ################################################## KNN ##################################################

   
   knn_output_path = './output/KNN1'
   knn = KNNModel(x_train, x_test, y_train, y_test, output_path=knn_output_path)

   model, pca = knn.PCASearch(ncomponents=list(range(1, 61)))

   print("Best Parameter by search")
   print("pca: ", pca)
   print(model)

   score, y_predict, y_proba = knn.test_model(model, pca)
   print('Acc: {}'.format(score))
   knn.evaluate(target_names)


   prediction_titles = [get_title(y_predict, y_test, target_names, i)
                     for i in range(y_predict.shape[0])]

   
   plot_gallery(x_test_plot, prediction_titles, h, w,  save_path=knn_output_path)

   



    

