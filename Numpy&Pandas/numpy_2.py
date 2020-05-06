'''
Numpy的索引1
import numpy as np 
A = np.arange(3,15).reshape((3,4))
print(A)
print(A[2])		#索引二维数组中的第二行，实际为第三行
print(A[2][1])		#索引二维数组中第二行第一列，实际为第三行第二列
print(A[2,1])		#同上
print(A[2:])		#索引第二行所有数
print(A[1, 1:3])	#索引第一行，第一列和第二列（冒号前包括，冒号后不包括）
'''

'''
#Numpy的索引2
import numpy as np
A = np.arange(3,15).reshape((3,4))
print(A)
print(A.T)
print(A.flatten())		#flatten将二维数组展开成一行的数列
for item in A.flat:		#flat是一个迭代器
	print(item)			#打印数列中的每个元素
'''

'''
#Numpy中array的合并
import numpy as np
A = np.array([1,1,1])
B = np.array([2,2,2])

C = np.vstack((A,B))		#上下合并两个array
D = np.hstack((A,B))		#左右合并两个array
A_1 = A[:, np.newaxis]		#冒号在np.newaxis前表示行增加维度，冒号在np.newaxis后表示列增加维度
B_1 = B[:, np.newaxis]
print(np.hstack((A_1,B_1)))
print(np.vstack((A,B,A,B)))
print(A_1)
E = np.concatenate((A_1,B_1,A_1,B_1), axis=1)		#指定维度合并，axis=0为上下合并，axis=1为左右合并
print(E)

#例子
import numpy as np
A = np.array([[1,1],
			  [1,1],
			  [1,1]])
B = np.array([[2,3],
			  [2,3],
			  [2,3]])
print(np.concatenate((A,B,A,B), axis=0))
'''

'''
#Numpy中array的分割
import numpy as np

A = np.arange(12).reshape((3,4))
print(A)

B = np.split(A, 2, axis=1)		#axis=1（按列分割）横向分割，将之前四列的二维数组分成两个两列的二维数组
b = np.hsplit(A, 2)	#同上
C = np.split(A, 3, axis=0)		#axis=0（按行分割）纵向分割，将之前三行的二维数组分成三个一行的数组
c = np.vsplit(A, 3)	#同上
D = np.array_split(A, 3, axis=1)		#不等量分割，四列分成2：1：1

print(b)
print(c)
'''

'''
#Numpy copy
import numpy as np

a = np.arange(4)
b = a   		#将a的值完全赋值给b，a不管什么时候改变b都会改变
c = a
d = b
a[0] = 11
d[1:3] = [22,33]

print(a)
'''

#Numpy中的deep copy
import numpy as np
a = np.arange(4)
a[0] = 322
b = a.copy()		#将a的值deep copy给b，如果之后a发生改变，b不再改变
a[1] = 333
print(a,b)
