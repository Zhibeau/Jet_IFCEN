import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedKFold
import gc
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

# 设置显示行列的最大值
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# 读取数据
url1 = "simple_train_R04_jet.csv"
url2 = "simple_test_R04_jet.csv"
train_csv = pd.read_csv(url1)
test_csv = pd.read_csv(url2)


# 数据处理函数
class Data:
    d = {1: [1, 0, 0, 0, ], 4: [0, 1, 0, 0], 5: [0, 0, 1, 0], 21: [0, 0, 0, 1]}

    def __init__(self, df):
        self.df = df
        self.train_x = self.df.values
        self.train_y = self.df.values

    def base(self):
        self.df['distance'] = self.df.apply(
            lambda df: (df['jet_px'] ** 2 + df['jet_py'] ** 2 + df['jet_pz'] ** 2) ** 0.5, axis=1)
        self.df['x_n'] = self.df['jet_px'] / self.df['distance']
        self.df['y_n'] = self.df['jet_py'] / self.df['distance']
        self.df['z_n'] = self.df['jet_pz'] / self.df['distance']

    def x_get(self):
        columns = []
        for c in list(self.df.columns):
            if c not in ['label', 'jet_id', 'event_id']:
                columns.append(c)
        print("总共" + str(len(columns)) + "个维度")
        return self.df.loc[:, columns].values

    def y_get(self):
        d = {1: 0, 4: 1, 5: 2, 21: 3}
        return self.df['label'].apply(lambda x: d[x]).values

    # method={'sum', 'count', 'mean','std','max','min'}
    # method也可以使用自定义函数，更复杂的可以用多列数据 https://zhuanlan.zhihu.com/p/36297802
    def data_generate(self, base_column, edit_column, method, name=None):
        if not name:
            name = base_column + '_' + edit_column + '_' + method
        tp = self.df.groupby(base_column)[edit_column].agg(method).reset_index()
        tp.columns=[base_column,name]
        self.df=self.df.merge(tp, on=base_column, how='left')

    # 数据清洗，这里只给了IQR方法的例子，具体方法以及实现后续完善 https://zhuanlan.zhihu.com/p/76281678
    def data_cleaning(self, cleaning_column):
        cleaning_data = self.df[cleaning_column]
        iqr = np.quantile(cleaning_data, 0.75) - np.quantile(cleaning_data, 0.25)
        down = np.quantile(cleaning_data, 0.25) - 1.5 * iqr
        up = np.quantile(cleaning_data, 0.75) + 1.5 * iqr
        valid_values = [float(down), float(up)]
        return valid_values

    # 后续会添加更多方法

def df_generate(csv_file):
    data=Data(csv_file)
    data.base()
    # 主要通过编写data_generate来生成不同维度
    data.data_generate('event_id','x_n','sum')
    return data

# 得到需要用到的数据
train=df_generate(train_csv)
test=df_generate(test_csv)
train_x = train.x_get()
test_x = test.x_get()
train_y = train.y_get()





# 把数据带入模型部分
# fold = StratifiedKFold(n_splits=5, shuffle=False)
# results = np.zeros((len(test_x), 4), dtype='float')
# for train, valid in fold.split(train_x, train_y):
#     X_train = train_x[train]
#     X_valid = train_x[valid]
#     Y_train = train_y[train]
#     Y_valid = train_y[valid]
#     model = CatBoostClassifier(
#         iterations=1000,
#         od_type='Iter',
#         od_wait=120,
#         max_depth=8,
#         learning_rate=0.02,
#         l2_leaf_reg=9,
#         random_seed=2019,
#         metric_period=50,
#         fold_len_multiplier=1.1,
#         loss_function='MultiClass',
#         logging_level='Verbose'
#     )
#     model.fit(X_train, Y_train, eval_set=(X_valid, Y_valid), use_best_model=True)
#     valid_pred = model.predict_proba(X_valid)
#     valid_pred = valid_pred.argmax(axis=1)
#     results += model.predict_proba(test_x)
#     del model
#     gc.collect()
#
# sub = pd.DataFrame()
# pred = results.argmax(axis=1)
# dd = {0: 1, 1: 4, 2: 5, 3: 21}
# sub['label'] = list(pred)
# sub['label'] = sub['label'].apply(lambda x:dd[x])
# test_id = test.df.loc[:,['jet_id']]
# sub['id'] = list(test_id)
# sub.to_csv('sub.csv', index=False)
