'''
# sklearn cross-validation 交叉验证 ---1
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(knn.score(X_test, y_test))

# # 使用交叉验证对数据进行训练，交叉验证次数为5
# knn = KNeighborsClassifier(n_neighbors=5)		#knn算法中人工定义的k值，初始选择最邻近的数据点的数量
# scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
# print(scores.mean())

# 可视化knn过程
k_range = range(1, 31)
k_scores = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k)
	# scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')		#for classification
	loss = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error') 	#for regression
	k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
'''








'''
# sklearn cross-validation 交叉验证 ---2
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

digits = load_digits()
X = digits.data
y = digits.target

train_sizes, train_loss, test_loss = learning_curve(SVC(gamma=0.001), X, y, cv=10,
													scoring='neg_mean_squared_error',
													train_sizes=[0.1, 0.25, 0.5, 0.75, 1])
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color="r", label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g", label="Cross-Validation")

plt.xlabel("Training example")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
'''











# sklearn cross-validation 交叉验证 ---3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve

# 载入数据
digits = load_digits()
X = digits.data
y = digits.target
param_range = np.logspace(-6, -2.3, 5)
train_loss, test_loss = validation_curve(
	SVC(), 	X, y, param_name='gamma',
	param_range=param_range, cv=10,
	scoring='neg_mean_squared_error'
	)
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color="r", label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g", label="Cross-Validation")

# ylabel中loss指误差
plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()