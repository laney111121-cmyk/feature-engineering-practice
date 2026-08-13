import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pima=pd.read_csv('pima-indians-diabetes.csv',header=0,names=['times_pregnant', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness',
                    'serum_insulin', 'bmi', 'pedigree_function', 'age', 'onset_diabetes']
)
print(pima.head())
print(pima.shape)
pima.info()
print(pima.describe())

#bmi等最小值为0违背科学，应是将缺失值标记成了‘0’
columns = ['serum_insulin', 'bmi', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness']
for col in columns:
    pima[col].replace(0,np.nan,inplace=True)
pima.info()

#查看删除缺失值前后数据差距
pima_dropped=pima.dropna()
print(f"pima行数:{pima.shape[0]},pima_dropped行数:{pima_dropped.shape[0]}")
#空准确率
print(pima['onset_diabetes'].value_counts(normalize=True))
print(pima_dropped['onset_diabetes'].value_counts(normalize=True))
#均值
print(pima.mean())
print(pima_dropped.mean())
(pima.mean()-pima_dropped.mean()).plot(kind='bar')
plt.show()

#基准线,对pima_dropped进行机器学习获得准确率
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
X_dropped=pima_dropped.drop('onset_diabetes',axis=1)
X_dropped=pd.DataFrame(StandardScaler().fit_transform(X_dropped),columns=X_dropped.columns)
y_dropped=pima_dropped['onset_diabetes']
X_dropped_train,X_dropped_test,y_dropped_train,y_dropped_test=train_test_split(X_dropped,y_dropped,test_size=0.2,random_state=42)
knn=KNeighborsClassifier()
knn_params={'n_neighbors':[2,3,4,5,6,7,8,9,10]}
grid_dropped=GridSearchCV(knn,knn_params,cv=5,scoring='recall')
grid_dropped.fit(X_dropped_train,y_dropped_train)
print(grid_dropped.best_params_)
y_dropped_pred=grid_dropped.predict(X_dropped_test)
print(classification_report(y_dropped_test,y_dropped_pred))

#在机器学习流水线中填充值,比较用median/mean/其他参数填充出来的预测准确度/召回率有没有提高
"""
因为学习算法的目标是泛化训练集的模式并将其应用于测试集。如果在划分数据集和
应用算法之前直接对整个数据集填充值,我们就是在作弊,模型其实学不到任何模式
"""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
X_pipe=pima.drop('onset_diabetes',axis=1)
X_pipe=pd.DataFrame(StandardScaler().fit_transform(X_pipe),columns=X_pipe.columns)
y_pipe=pima['onset_diabetes']
X_pipe_train,X_pipe_test,y_pipe_train,y_pipe_test=train_test_split(X_pipe,y_pipe,test_size=0.2,random_state=42)
knn_params_pipe={'imputer__strategy':['median','mean'],'classify__n_neighbors':[2,3,4,5,6,7,8,9,10]}
knn=KNeighborsClassifier()
model_pipe=Pipeline([
    ('imputer',SimpleImputer()),#simpleimputer算法找最佳strategy
    ('classify',knn)#knn算法找最佳的params
])
grid_pipe=GridSearchCV(model_pipe,knn_params_pipe,cv=5,scoring='recall')#cv=5为交叉验证
grid_pipe.fit(X_pipe_train,y_pipe_train)
print(grid_pipe.best_params_)
y_pipe_pred=grid_pipe.predict(X_pipe_test)
print(classification_report(y_pipe_test,y_pipe_pred))
#knn算法极度依赖数值大小，故先对x部分进行归一化（注意y为0-1数值，不能归一化），从而用mean填充的准确率明显上升
#由于在医学上，漏诊比误诊更严重，所以我们选择高召回率，即使用均值填充数据集