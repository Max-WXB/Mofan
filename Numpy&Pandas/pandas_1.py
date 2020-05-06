'''
#pandas的基本介绍
import pandas as pd 
import numpy as np 
s = pd.Series([1,3,6,np.nan,44,1])
print(s)
dates = pd.date_range('20200414', periods=6)
print(dates)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['a','b','c','d'])		#index为行名，column为列名
print(df)
df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
print(df1)
df2 = pd.DataFrame({'A' : 1.,
					'B' : pd.Timestamp('20200414'),
					'C' : pd.Series(1, index=list(range(4)),dtype='float32'),
					'D' : np.array([3] * 4, dtype='int32'),
					'E' : pd.Categorical(["test","train","test","train"]),
					'F' : 'foo'})
df2_sort = df2.sort_index(axis=1, ascending=False)		#axis=1对列名进行排序，axis=0对行名进行排序，ascending=False为倒序
df2_sort2 = df2.sort_values(by='E')			#sort_index是对列或者行名进行排序，values是对某一列中的值进行排序
print(df2, df2.dtypes)
print(df2.describe())
print(df2.T)
print(df2_sort2)
'''

'''
#pandas选择数据
import pandas as pd
import numpy as np

dates = pd.date_range('20140202', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates,columns=['A','B','C','D'])
print(df['A'], '\n', df.A)		#选择名字为A的那一列
print(df[0:3], '\n', df['20140202':'20140204'])		#选择1到3行
print(df.loc['20140203'])		#利用行标签对数据进行选择
print(df.loc['20140202':'20140203', ['A','B']])		#利用loc选择，[]中逗号前是选择的行标题，逗号后是选择的列标题
print(df.iloc[3:5,1:2])			#利用iloc定位里面的值，第三行到第五行不包括第五行，第一列到第二列不包括第二列（：前包括，：后不包括）
print(df.iloc[[1,3,5],1:2])		#iloc可以不连续的筛选
print(df[df.A > 8])			#条件筛选，对A进行过滤，只显示A中值大于8的项目
'''


'''
#pandas设置值
import pandas as pd
import numpy as np

dates = pd.date_range('20140202', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates,columns=['A','B','C','D'])

df.iloc[2,2]  = np.nan			#选择其中第2行第2列，更改为nan（是从0开始数，[2,2]其实是第三行第三列）
df.loc['20140202', 'B'] = 233333			#选择行名为’20140202‘，列名为’B‘的数值进行更改
df.B[df.A>0] = 'Max'		#判断A列中，选中A列中大于0的值，更改B中这些行的值为Max
df['F'] = 'Hello'			#更改F列中所有值为Hello
df['E'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20140202',periods=6))			#新增一列E，但是列的长度必须与之前的长度对其
print(df)
'''


'''
#pandas处理丢失数据
import pandas as pd
import numpy as np

dates = pd.date_range('20140202', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates,columns=['A','B','C','D'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan

df2 = df.dropna(axis=1, how='any')		#axis=0删去有nan值的行，axis=1删去有nan值得列,how为any时只要有nan则删，how为all则必须这一行或列都为nan
df3 = df.fillna(value=0)		#替换nan的值为value
df4 = df.isnull()			#判断里面是否有缺失值，false为无，true为有
print(np.any(df.isnull()) == True)		#判断俩面是否有缺失值
print(df4)
'''

