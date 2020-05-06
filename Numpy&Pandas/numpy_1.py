'''
Numpy的属性
import numpy as np
array = np.array([[1,2,3],
				  [2,3,4]])
print(array)
print('number of dim:', array.ndim)
print('shape:', array.shape)
print('size:', array.size)
'''

'''
#Numpy创建array
import numpy as np 
a = np.array([1,2,3])
b = np.array([[1,2,3],
			  [2,3,4]])
c = np.zeros((3,4))
d = np.ones((4,6), dtype = np.int)
e = np.empty((3,4))
f = np.arange(10,20,2)
g = np.arange(12).reshape((3,4))
h = np.linspace(1,10,6).reshape((2,3))

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)
'''


'''
#Numpy的基础运算1
import numpy as np
a = np.array([[1,1],
			  [0,1]])
b = np.arange(4).reshape((2,2))
print(a)
print(b)
c = a*b		#逐个相乘
c_dot = np.dot(a, b)	#矩阵运算
c_dot_2 = a.dot(b)		#矩阵运算方法2
print(c, c_dot, c_dot_2)
'''

'''
#Numpy的基础运算2
import numpy as np
a = np.random.random((2,4))
print(a)
print(np.sum(a, axis=1))	#axis=1列中求和，axis=0行中求和
print(np.min(a, axis=0))
print(np.max(a, axis=1))
'''

#Numpy的基础运算3
import numpy as np
A = np.arange(2,14).reshape((3,4))
print(np.argmin(A))		#索引矩阵中最小值得位置
print(np.argmax(A))		#索引矩阵中最大值的位置
print(np.mean(A))
print(A.mean())		#计算平均值的两种方法
print(np.median(A))
print(np.cumsum(A))		#累加过程，前n项的和
print(np.diff(A))		#累差，n-(n-1)
print(np.nonzero(A))	#输出非零数的位置，第一个数组是行数，第二个数组是列数
print(np.sort(A))		#逐行排序
print(np.transpose(A))	#矩阵的转置，行变成列，列变成行
print((A.T).dot(A))		#矩阵的转置2
print(np.clip(A,5,9))	#小于5的等于5，大于9的等于9
print(np.mean(A,axis=0))	#axis=0对于列进行计算平均值，axis=1对于行进行计算平均值

