# save保存model
from sklearn import svm 
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X = iris.data 
y = iris.target
clf.fit(X, y)

# # method_1 pickle
# import pickle
# # save
# with open('save/clf.pickle', 'wb') as f:
# 	pickle.dump(clf, f)
# # restore
# with open('save/clf.pickle', 'rb') as f:
# 	clf2 = pickle.load(f)
# 	print(clf2.predict(X[0:1]))

# method_2 joblib
import joblib
# save 
joblib.dump(clf, 'save/clf.pkl')
# restore
# clf3 = joblib.load('save/clf.pkl')
# print(clf3.predict(X[0:1]))

