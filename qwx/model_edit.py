import numpy as np
from catboost import CatBoostClassifier
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold


class Model:
    def __init__(self, train_x, train_y, test_x):
        self.X = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.y = self.train_y.argmax(axis=1)
        self.mean_fpr = np.linspace(0, 1, 100)

    # 下面先分类数据集
    # 再按照分好的数据集训练
    # 最后做计算ROC和AUC来做验证
    def fit(self):
        tprs = []
        aucs = []
        cv = StratifiedKFold(n_splits=3)
        results = np.zeros((len(self.test_x), 4), dtype='float')
        classifier = CatBoostClassifier(
            iterations=10,
            od_type='Iter',
            od_wait=120,
            max_depth=8,
            learning_rate=0.02,
            l2_leaf_reg=9,
            random_seed=2020,
            metric_period=50,
            fold_len_multiplier=1.1,
            loss_function='MultiClass',
            logging_level='Verbose')

        for i, (train, test) in enumerate(cv.split(self.X, self.y)):
            '''请在这里重新定义一次分类器，使其中参数刷新'''
            '''再用第i种数据集分类方法训练'''
            classifier.fit(self.X[train], self.y[train])
            '''下面开始验证'''
            y_pred = classifier.predict_proba(self.X[test])
            y_test = np.reshape(self.train_y[test], (-1, 1))
            valid_pred = np.reshape(y_pred, (-1, 1))
            fpr, tpr, _ = roc_curve(y_test, valid_pred)
            aucc = auc(fpr, tpr)
            interp_tpr = interp1d(fpr, tpr, kind='linear')
            interp_tpr = interp_tpr(self.mean_fpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(aucc)
        return tprs

    def auc_get(self):
        tprs = self.fit()
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        return mean_auc
