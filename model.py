import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from visualization import plot_curve

from metrics import draw_ROC, model_evaluation, show_classification_report
import utils


class MLModel:
    def __init__(self, x_train, x_test, y_train, y_test, draw_roc=False, output_path=None, **kwargs):

        self.x_train, self.x_test, self.y_train, self.y_test =  x_train, x_test, y_train, y_test
        self.model = None
        self.y_predict = None
        self.y_proba = None
        self.output_path = output_path

        self.draw_roc = draw_roc
        self.draw_roc = True

        utils.exist_make(self.output_path)

    def train_model(self):
        return self.model

    def test_model(self,):
        pass

    def evaluate(self, target_names):
        # print('In evaluate {}'.format(os.path.join(self.output_path, 'ROC.jpg')))
        # if self.draw_roc: draw_ROC(y_test=self.y_test, y_proba=self.y_proba, output_path=os.path.join(self.output_path, 'ROC.jpg'))
        model_evaluation(self.y_test, self.y_predict)
        show_classification_report(self.y_test, self.y_predict, target_names)


class PCAModel(MLModel):
    def __init__(self, x_train, x_test, y_train, y_test, pca_ncomponents=0, output_path='./output/SVM'):
        MLModel.__init__(self, x_train, x_test, y_train, y_test, output_path)

        self.model = GridSearchCV(PCA(n_components=self.pca_ncomponents), self.param_grid, n_jobs=-1, cv=5)

    def train_model(self):

        self.x_train = self.model.fit_transform(self.x_train)

        # print(self.model.explained_variance_)
        # print(self.model.explained_variance_ratio_)
        # print(len(self.model.explained_variance_ratio_))


        return self.model.explained_variance_ratio_


    def test_model(self,):
        pass



class SVMModel(MLModel):
    def __init__(self, x_train, x_test, y_train, y_test, draw_roc=False, svm_ncomponents=0, output_path='./output/SVM'):
        MLModel.__init__(self, x_train, x_test, y_train, y_test, draw_roc=draw_roc, output_path=output_path)

        self.svm_ncomponents = svm_ncomponents
        self.param_grid = {'C': np.linspace(1, 1e3, 50),
                            'gamma': np.linspace(1e-3, 1, 50), 
                            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                            }

        self.grid_search = GridSearchCV(SVC(class_weight='balanced'), self.param_grid, n_jobs=-1, cv=5)


    def PCASearch(self, ncomponents: list):
        train_list, test_list, score_list = [], [], []
        best_model, best_score = None, 0
        for components in ncomponents:
            print('components:', components)
            model, train_score = self.train_model(components)
            if best_score < train_score: 
                best_score = train_score
                best_model = best_model
            test_score, y_predict, y_proba = self.test_model()
            train_list.append(train_score)
            test_list.append(test_score)
        
        score_list.append(train_list)
        score_list.append(test_list)

        max_score = np.array([[max(train_list)], [max(test_list)]])
        max_idx = np.array([[ncomponents[train_list.index(max(train_list))]], [ncomponents[test_list.index(max(test_list))]]])
        

        x, y = np.array([ncomponents]*2), np.array(score_list)
        plot_curve(x=x, y=y, idxs=max_idx, 
                    values=max_score, xlabel='PCA components number', 
                    xlim=(1, max(x.ravel())+1), ylim=(0, max(y.ravel())*1.02), 
                    ylabel='Score', output_dir=os.path.join(self.output_path, 'KNN_PCA.jpg'))

        self.model = model # find best model by search


    def train_model(self, svm_ncomponents):
        
        x_train = self.x_train
        self.svm_ncomponents = svm_ncomponents
        if self.svm_ncomponents:
            self.pca = PCA(n_components=self.svm_ncomponents)
            x_train = self.pca.fit_transform(x_train)
        
        self.grid_search.fit(x_train, self.y_train)
        train_score = self.grid_search.best_score_
        print("Best estimator found by grid search:")
        print("Best params: ", self.grid_search.best_params_)


        self.model = self.grid_search.best_estimator_

        return self.model, train_score


    def test_model(self,):
        x_test = self.x_test
        if self.svm_ncomponents:
            x_test = self.pca.transform(x_test)
        
        self.y_predict = self.model.predict(x_test)
        self.y_proba = self.model.decision_function(x_test)
        score = self.model.score(x_test, self.y_test)
        return score, self.y_predict, self.y_proba



class KNNModel(MLModel):
    def __init__(self, x_train, x_test, y_train, y_test, draw_roc=False, knn_ncomponents=-1, output_path='./output/KNN'):
        MLModel.__init__(self, x_train, x_test, y_train, y_test, draw_roc=draw_roc, output_path=output_path)
        
        self.knn_ncomponents = knn_ncomponents
        self.param_grid = [
                            {
                                'weights':['uniform'],
                                'n_neighbors':[i for i in range(5, 50)]
                            },
                            {
                                'weights':['distance'],
                                'n_neighbors':[i for i in range(5, 50)],
                                'p':[i for i in range(1, 6)]
                            }
                        ]

        self.grid_search = GridSearchCV(KNeighborsClassifier(), self.param_grid, n_jobs=-1, cv=5)
    


    def PCASearch(self, ncomponents: list):
        
        train_list, test_list, score_list = [], [], []
        best_model, best_score = None, 0
        for components in ncomponents:
            print('components:', components)
            model, train_score = self.train_model(components)
            if best_score < train_score: 
                best_score = train_score
                best_model = best_model
            test_score, y_predict, y_proba = self.test_model()
            train_list.append(train_score)
            test_list.append(test_score)
        
        score_list.append(train_list)
        score_list.append(test_list)

        max_score = np.array([[max(train_list)], [max(test_list)]])
        max_idx = np.array([[ncomponents[train_list.index(max(train_list))]], [ncomponents[test_list.index(max(test_list))]]])
        
        x, y = np.array([ncomponents]*2), np.array(score_list)
        plot_curve(x=x, y=y, idxs=max_idx, 
                    values=max_score, xlabel='PCA components number', 
                    xlim=(1, max(x.ravel())+1), ylim=(0, max(y.ravel())*1.02), 
                    ylabel='Score', output_dir=os.path.join(self.output_path, 'KNN_PCA.jpg'))

        self.model = model # find best model by search

    def train_model(self, knn_ncomponents):

        x_train = self.x_train
        self.knn_ncomponents = knn_ncomponents
        if self.knn_ncomponents:
            self.pca = PCA(n_components=self.knn_ncomponents)
            x_train = self.pca.fit_transform(x_train)
        
        self.grid_search.fit(x_train, self.y_train)
        train_score = self.grid_search.best_score_
        print("Best estimator found by grid search:")
        print("Best params: ", self.grid_search.best_params_)


        self.model = self.grid_search.best_estimator_

        return self.model, train_score


    def test_model(self,):
        x_test = self.x_test
        if self.knn_ncomponents:
            x_test = self.pca.transform(x_test)

        self.y_predict = self.model.predict(x_test)
        self.y_proba = self.model.predict_proba(x_test)
        score = self.model.score(x_test, self.y_test)
        return score, self.y_predict, self.y_proba







