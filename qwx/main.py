import pandas as pd

from data_edit import Data
from model_edit import Model


def df_generate(csv_file):
    df = Data(csv_file)
    df.base()
    # 主要通过编写data_generate来生成不同维度 重要
    for p in ['x_n', 'y_n', 'z_n']:
        df.data_generate('event_id', p, 'sum')
    for p in ['number_of_particles_in_this_jet', 'jet_mass']:
        for m in ['sum', 'min', 'max']:
            df.data_generate('event_id', p, m)
    df.data_ratio('jet_energy', 'number_of_particles_in_this_jet', 'mean_energy')
    df.data_ratio('jet_mass', 'number_of_particles_in_this_jet', 'mean_jet_mass')
    for p in ['mean_energy', 'mean_jet_mass']:
        for m in ['sum', 'min', 'max']:
            df.data_generate('event_id', p, m)
    # df.df['new'] = df.df['jet_energy'] * (df.df['distance'] * df.df['distance'])
    # df.data_multiply('jet_energy', 'distance')
    # df.data_delete()
    return df


# 得到需要用到的数据
url1 = "simple_train_R04_jet.csv"
url2 = "simple_test_R04_jet.csv"
train_csv = pd.read_csv(url1)
test_csv = pd.read_csv(url2)
train = df_generate(train_csv)
test = df_generate(test_csv)
columns, n = train.declare()

train_x = train.x_get()
test_x = test.x_get()
train_y = train.y_get()

# 开始拟合
model = Model(train_x, train_y, test_x)
# 得到auc
auc = model.auc_get()

print(auc)

with open('total.txt', 'a+') as f:
    f.write(str(auc) + ',' + n + ',' + str(columns) + '\n')
