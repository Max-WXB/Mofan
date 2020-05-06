'''
#pandas导入导出
import pandas as pd 

data = pd.read_csv('test.csv')
print(data)

data.to_pickle('test.pickle')
'''


'''
#pandas 合并concat_1
import pandas as pd 
import numpy as np 

df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])

res = pd.concat([df1,df2,df3],axis=0, ignore_index=True)			#axis=0是纵向的合并（增加行），axis=1是横向合并（增加列），ignore_index=True重新命名行名
print(res)
'''


'''
#pandas 合并concat_2
import pandas as pd 
import numpy as np 

df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])

print(df1)
print(df2)

res = pd.concat([df1,df2], join='outer', ignore_index=True)			#join='outer'时显示两者所有的行和列，没有数值的位置用nan补充。join='inner'时只显示两者共有的部分
print(res)
'''


'''
#pandas 合并concat_3
import pandas as pd 
import numpy as np 

df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])

res = pd.concat([df1,df2], axis=1, join_axes=[df2.index])		#join_axes可以选择按照哪个dataframe进行合并，合并后的行列与选择的dataframe一致
print(res)
'''

'''
#pandas 合并concat_4
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
print(s1)
res = df1.append([df2,df3], ignore_index=True)			#append只能纵向增加（增加行），
res1 = df1.append(s1, ignore_index=True)
print(res1)
'''


'''
#pandas 合并merge_1
import pandas as pd
import numpy as np

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
print(left)
print(right)
res = pd.merge(left, right, on='key')			#单个key时，on='key'，基于key列进行合并
print(res)
'''


'''
#pandas 合并merge_2
import pandas as pd

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                      'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                       'key2': ['K0', 'K0', 'K0', 'K0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})
print(left)
print(right)
res = pd.merge(left, right, on=['key1','key2'], how='outer')			#根据两个key进行合并，默认根据inner，只考虑相同部分
#how有四种方法left，right，inner，outer，inner只考虑形同部分，outer不考虑是否相同，left和right只基于其中一组数据进行合并。
print(res)
'''


'''
#pandas 合并merge_3
import pandas as pd

#定义资料集并打印出
df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
print(df1)
print(df2)

res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
res1 = pd.merge(df1, df2, on='col1', how='outer', indicator=False)		#indicator可以显示数据是从哪个原始数据中来的，默认为false
print(res1)
'''

'''
#pandas 合并merge_4
import pandas as pd
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])
print(left)
print(right)
res = pd.merge(left, right, left_index=True, right_index=True, how='outer')			#根据index进行合并，index就是行名，根据行名将相同的合并
print(res)
'''



#pandas 合并merge_5
import pandas as pd

#定义资料集
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})

res = pd.merge(boys,girls, on='k', suffixes=['_boys','_girls'], how='inner')			#suffinxes可以为数据增加后缀名进行区分后合并
print(res)





