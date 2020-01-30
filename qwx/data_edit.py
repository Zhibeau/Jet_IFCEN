import numpy as np
import pandas as pd

# 设置显示行列的最大值
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# 读取数据


# 数据处理函数
class Data:

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
        return self.df.loc[:, columns].values

    def y_get(self):
        d = {1: [1, 0, 0, 0, ], 4: [0, 1, 0, 0], 5: [0, 0, 1, 0], 21: [0, 0, 0, 1]}
        return np.array(list(self.df['label'].apply(lambda x: d[x]).values))

    # method={'sum', 'count', 'mean','std','max','min'}
    # method也可以使用自定义函数，更复杂的可以用多列数据 https://zhuanlan.zhihu.com/p/36297802
    def data_generate(self, base_column, edit_column, method, name=None):
        if not name:
            name = base_column + '_' + edit_column + '_' + method
        tp = self.df.groupby(base_column)[edit_column].agg(method).reset_index()
        tp.columns = [base_column, name]
        self.df = self.df.merge(tp, on=base_column, how='left')

    def data_ratio(self, column1, column2, name=None):
        if not name:
            name = column1 + '/' + column2
        self.df[name] = self.df[column1] / self.df[column2]

    def data_multiply(self, column1, column2, name=None):
        if not name:
            name = column1 + '*' + column2
        self.df[name] = self.df[column1] * self.df[column2]

    def data_delete(self, column):
        del self.df[column]

    def declare(self):
        columns = []
        for c in list(self.df.columns):
            if c not in ['label', 'jet_id', 'event_id']:
                columns.append(c)
        print(columns)
        print("总共以上" + str(len(columns)) + "个维度")
        return columns, str(len(columns))

    # 数据清洗，这里只给了IQR方法的例子，具体方法以及实现后续完善 https://zhuanlan.zhihu.com/p/76281678
    def data_cleaning(self, cleaning_column):
        cleaning_data = self.df[cleaning_column]
        iqr = np.quantile(cleaning_data, 0.75) - np.quantile(cleaning_data, 0.25)
        down = np.quantile(cleaning_data, 0.25) - 1.5 * iqr
        up = np.quantile(cleaning_data, 0.75) + 1.5 * iqr
        valid_values = [float(down), float(up)]
        return valid_values
