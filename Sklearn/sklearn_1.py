'''
# 用sklearn进行学习的基础模式
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 引用iris数据，X是分类标签，y是分类目标
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# print(iris_X[:2,:])
# print(iris_y)

# 将数据集分成训练集和测试集，成分比例为7：3，分成两个数据集后将数据打乱
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
# print(y_train)

# 定义使用sklearn中使用的哪个模块
knn = KNeighborsClassifier()
# fit是将数据进行training
knn.fit(X_train, y_train)

print(knn.predict(X_test))
print(y_test)
'''










'''
# sklearn数据库使用
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

# # 载入波士顿房价信息
# loaded_data = datasets.load_boston()
# data_X = loaded_data.data
# data_y =loaded_data.target

# model = LinearRegression()
# # fit是使model学习LinearRegression方法
# model.fit(data_X, data_y)

# print(model.predict(data_X[:4,:]))
# print(data_y[:4])

# 使用sklearn中的make_regression自己制造数据
X, y = datasets.make_regression(n_samples=100, n_features=1,n_targets=1, noise=8)
plt.scatter(X, y)
plt.show()
'''








'''
# sklearn 常用属性与功能
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# 载入波士顿房价信息
loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y =loaded_data.target

model = LinearRegression()
# fit是使model学习LinearRegression方法
model.fit(data_X, data_y)

# print(model.coef_)			#y=0.1x+0.3
# print(model.intercept_)		#函数和y轴交点
# print(model.get_params())
print(model.score(data_X, data_y))			#对model学到的东西打分，看是否吻合，打分标准：R^2 coefficient of determination
'''








'''
# sklearn 中的normalization 标准化数据
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# a = np.array([[10, 2.7, 3.6],
# 			  [-100, 5,-2],
# 			  [210, 20, 40]], dtype=np.float64)

# print(a)
# print(preprocessing.scale(a))

# 生成数据
X, y = make_classification(n_samples=500,		#样本个数
							n_features=2,		#特征个数
							n_redundant=0,		#冗余信息
							n_informative=2,	#多信息特征的个数
							random_state=22,	#随机数生成器
							n_classes=2,		#类别个数
							n_clusters_per_class=1, #某一个类别是由几个cluster构成的
							scale=100)	
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC()
clf.fit(X_train,y_train)
print(clf.score(X_test, y_test))
'''













