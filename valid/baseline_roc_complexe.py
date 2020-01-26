import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import gc
import os
from sklearn.metrics import classification_report
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBClassifier
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
def count_column(df,column):
    tp = df.groupby(column).count().reset_index()
    tp = tp[list(tp.columns)[0:2]]
    tp.columns = [column, column+'_count']
    df=df.merge(tp,on=column,how='left')
    return df
def count_mean(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['mean']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_mean']
    df = df.merge(tp, on=base_column, how='left')
    return df
def count_count(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['count']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_count']
    df = df.merge(tp, on=base_column, how='left')
    return df
def count_sum(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['sum']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_sum']
    df = df.merge(tp, on=base_column, how='left')
    return df
def count_std(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['std']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_std']
    df = df.merge(tp, on=base_column, how='left')
    return df



def energy(df):
    x=df['jet_px']
    y=df['jet_py']
    z= df['jet_pz']
    return (x**2+y**2+z**2)**0.5
pass
def pro_data():
    train=pd.read_csv('C:/Users/zzhib/Desktop/particles/jet_complex_data/complex_train_R04_jet.csv')
    test=pd.read_csv('C:/Users/zzhib/Desktop/particles/jet_complex_data/complex_test_R04_jet.csv')
    train['energy']=train.apply(energy,axis=1)
    test['energy']=test.apply(energy,axis=1)


    train['x_n']=train['jet_px']/train['energy']
    train['y_n']=train['jet_py']/train['energy']
    train['z_n']=train['jet_pz']/train['energy']

    test['x_n']=test['jet_px']/test['energy']
    test['y_n']=test['jet_py']/test['energy']
    test['z_n']=test['jet_pz']/test['energy']




    train=count_mean(train,'event_id','x_n')
    train=count_sum(train,'event_id','x_n')
    train=count_std(train,'event_id','x_n')

    train=count_mean(train,'event_id','y_n')
    train=count_sum(train,'event_id','y_n')
    train=count_std(train,'event_id','y_n')

    train=count_mean(train,'event_id','z_n')
    train=count_sum(train,'event_id','z_n')
    train=count_std(train,'event_id','z_n')


    test=count_mean(test,'event_id','x_n')
    test=count_sum(test,'event_id','x_n')
    test=count_std(test,'event_id','x_n')

    test=count_mean(test,'event_id','y_n')
    test=count_sum(test,'event_id','y_n')
    test=count_std(test,'event_id','y_n')

    test=count_mean(test,'event_id','z_n')
    test=count_sum(test,'event_id','z_n')
    test=count_std(test,'event_id','z_n')


    train['abs']=train['jet_energy']-train['energy']
    test['abs']=test['jet_energy']-test['energy']


    train=count_mean(train,'event_id','number_of_particles_in_this_jet')
    train=count_sum(train,'event_id','number_of_particles_in_this_jet')
    train=count_std(train,'event_id','number_of_particles_in_this_jet')

    train=count_mean(train,'event_id','jet_mass')
    train=count_sum(train,'event_id','jet_mass')
    train=count_std(train,'event_id','jet_mass')

    train=count_mean(train,'event_id','jet_energy')
    train=count_sum(train,'event_id','jet_energy')
    train=count_std(train,'event_id','jet_energy')

    train['mean_energy']=train['jet_energy']/train['number_of_particles_in_this_jet']
    train['mean_jet_mass']=train['jet_mass']/train['number_of_particles_in_this_jet']
    train=count_mean(train,'event_id','mean_energy')
    train=count_sum(train,'event_id','mean_energy')
    train=count_std(train,'event_id','mean_energy')
    train=count_mean(train,'event_id','mean_jet_mass')
    train=count_sum(train,'event_id','mean_jet_mass')
    train=count_std(train,'event_id','mean_jet_mass')
    train=count_mean(train,'event_id','abs')
    train=count_sum(train,'event_id','abs')
    train=count_std(train,'event_id','abs')
    train=count_mean(train,'event_id','energy')
    train=count_sum(train,'event_id','energy')
    train=count_std(train,'event_id','energy')







    test=count_mean(test,'event_id','number_of_particles_in_this_jet')
    test=count_sum(test,'event_id','number_of_particles_in_this_jet')
    test=count_std(test,'event_id','number_of_particles_in_this_jet')

    test=count_mean(test,'event_id','jet_mass')
    test=count_sum(test,'event_id','jet_mass')
    test=count_std(test,'event_id','jet_mass')

    test=count_mean(test,'event_id','jet_energy')
    test=count_sum(test,'event_id','jet_energy')
    test=count_std(test,'event_id','jet_energy')




    test['mean_energy']=test['jet_energy']/test['number_of_particles_in_this_jet']
    test['mean_jet_mass']=test['jet_mass']/test['number_of_particles_in_this_jet']
    test=count_mean(test,'event_id','mean_energy')
    test=count_sum(test,'event_id','mean_energy')
    test=count_std(test,'event_id','mean_energy')
    test=count_mean(test,'event_id','mean_jet_mass')
    test=count_sum(test,'event_id','mean_jet_mass')
    test=count_std(test,'event_id','mean_jet_mass')
    test=count_mean(test,'event_id','abs')
    test=count_sum(test,'event_id','abs')
    test=count_std(test,'event_id','abs')
    test=count_mean(test,'event_id','energy')
    test=count_sum(test,'event_id','energy')
    test=count_std(test,'event_id','energy')

    train=train.drop_duplicates(subset=['event_id']).reset_index(drop=True)
    with open('train_complex.pkl', 'wb') as file:
        pickle.dump(train, file)
    with open('test_complex.pkl', 'wb') as file2:
        pickle.dump(test, file2)
pro_data()

#############################导入数据#####################################
with open('train_complex.pkl', 'rb') as file:
    train = pickle.load(file)
with open('test_complex.pkl', 'rb') as file2:
    test = pickle.load(file2)

d={1:[1,0,0,0,],4:[0,1,0,0],5:[0,0,1,0],21:[0,0,0,1]}
def label_process(x):
    x=d[x]
    return x

train['label']=train['label'].apply(label_process)
train_y=train.pop('label').values
train_y=np.array(list(train_y))
_=train.pop('jet_id')
test_id=test.pop('jet_id')
_=train.pop('event_id')
_=test.pop('event_id')
train_x=train.values
test_x=test.values
X = train_x
y = train_y.argmax(axis=1)
##########################################################################

#下面先分类数据集
#再按照分好的数据集训练
#最后做计算ROC和AUC来做验证
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
cv = StratifiedKFold(n_splits=3)
results = np.zeros((len(test_x), 4), dtype='float')

for i, (train, test) in enumerate(cv.split(X, y)):
    '''请在这里重新定义一次分类器，使其中参数刷新'''
    classifier = CatBoostClassifier(
        iterations=1000,
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
    '''再用第i种数据集分类方法训练'''
    classifier.fit(X[train], y[train])
    '''下面开始验证'''
    y_pred = classifier.predict_proba(X[test])
    y_test = np.reshape(train_y[test],(-1,1))
    y_pred = np.reshape(y_pred,(-1,1))
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    aucc = auc(fpr, tpr)
    ax.plot(fpr,tpr, alpha=0.3, lw=1,label = 'ROC fold %d (AUC = %.2f)'%(i,aucc))
    interp_tpr = interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(aucc)    
    results += classifier.predict_proba(test_x)
    del classifier
    gc.collect()

sub=pd.DataFrame()
pred=results.argmax(axis=1)
dd={0:1,1:4,2:5,3:21}
def sub_process(x):
    x=dd[x]
    return x
sub['label']=list(pred)
sub['label']=sub['label'].apply(sub_process)
sub['id']=list(test_id)
sub.to_csv('sub_complex2.csv',index=False)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic")
ax.legend(loc="lower right")
plt.savefig("roc_complex2.png")
plt.show()

