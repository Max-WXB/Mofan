'''
#matplotlip 基础
import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(-1, 1, 50)
#y = 2*x+1
y = x**2
plt.plot(x, y)
plt.show()
'''

'''
#figure图像
import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(-3, 3, 50)
y1 = 2*x+1
y2 = x**2

plt.figure(figsize=(5,3))
plt.plot(x, y1)

plt.figure(num=3, figsize=(5,3))
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=3.0, linestyle='-.')
#color是颜色，linewidth是粗细，linestyle是线的风格，'--'是指虚线
#linestyle包含有'-','--','-.',':','None',' ','','solid','dashed','dashdot','dotted'
plt.show()
'''



'''
#设置坐标轴_1
import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(-3, 3, 50)
y1 = 2*x+1
y2 = x**2

plt.figure(figsize=(10,5))
plt.plot(x,y2)
plt.plot(x,y1, color='red', linewidth=2.0, linestyle='--')

plt.xlim((-1,2))		#设置x的取值范围为-1~2
plt.ylim((-2,3))		#设置y的取值范围为-2~3
plt.xlabel('I am x')	#描述x轴
plt.ylabel('I am y')	#描述y轴

new_ticks = np.linspace(-1, 2, 5)
print(new_ticks)
plt.xticks(new_ticks)		#更换x轴的单位大小
plt.yticks([-2, -1.8, -1, 1.22, 3],
			[r'$really\ bad$', r'$bad\ \alpha$', r'$normal$', r'$good$', r'$really\ good$'])
#将y轴上的数字与字符串对应，利用正则表达式表示文字和特殊符号
plt.show()
'''


'''
#设置坐标轴_2
import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(-3, 3, 50)
y1 = 2*x+1
y2 = x**2

plt.figure(figsize=(10,5))
plt.plot(x,y2)
plt.plot(x,y1, color='red', linewidth=2.0, linestyle='--')

plt.xlim((-1,2))		#设置x的取值范围为-1~2
plt.ylim((-2,3))		#设置y的取值范围为-2~3
plt.xlabel('I am x')	#描述x轴
plt.ylabel('I am y')	#描述y轴

new_ticks = np.linspace(-1, 2, 5)
print(new_ticks)
plt.xticks(new_ticks)		#更换x轴的单位大小
plt.yticks([-2, -1.8, -1, 1.22, 3],
			[r'$really\ bad$', r'$bad\ \alpha$', r'$normal$', r'$good$', r'$really\ good$'])
#将y轴上的数字与字符串对应，利用正则表达式表示文字和特殊符号
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')		#代替旧的x轴
ax.yaxis.set_ticks_position('left')			#代替旧的y轴
ax.spines['bottom'].set_position(('data', 0))		#.set_position设置边框位置
ax.spines['left'].set_position(('data', 0))			#.spines设置边框
#ax.xaxis.set_ticks_position所属位置有（top，bottom，both，default， none）
#.spines所属位置有（outward， axes，data）
plt.show()
'''

'''
#Legend图例
import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(-3, 3, 50)
y1 = 2*x+1
y2 = x**2

plt.figure(figsize=(10,5))
plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('I am x')
plt.ylabel('I am y')
new_ticks = np.linspace(-1, 2, 5)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2, -1.8, -1, 1.22, 3],
			[r'$really\ bad$', r'$bad\ \alpha$', r'$normal$', r'$good$', r'$really\ good$'])

l1, = plt.plot(x,y2, label='up')		#设置第一个函数图像名字为'up'
l2, = plt.plot(x,y1, color='red', linewidth=2.0, linestyle='--', label='down')		#设置第二个函数图像名字为'down'

plt.legend(handles=[l1,l2], labels=['up','down'], loc='best')
#创建图例，handles是指将图形封装进一个project，labels是为图例创建名称，loc是选择图例的显示位置，当loc选择best时图例会出现在图片中的空白位置
plt.show()
'''


'''
#Annotation 标注
import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(-3, 3, 50)
y = 2*x+1

plt.figure(num=1, figsize=(8, 5))
plt.plot(x, y, lw=3)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))


x0 = 1
y0 = 2*x0 + 1
plt.scatter(x0, y0, s=50, color='b')
plt.plot([x0,x0],[y0,0],'k--', lw=2.5)			#画辅助线，x0到x轴
plt.plot([x0,0],[y0,y0],'k--', lw=2.5)			#画辅助线，y0到y轴

plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2", color='green', lw=2.5))
#画其他辅助图，xycoords='data'基于数据的值选择位置，xytext=(+30,-30)，textcoords='offset points'对于标注位置的描述和xy的偏差值，arrowprops是对图中箭头类型的设置
plt.text(-3.7,3,r'$This\ is\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',
	fontdict={'size':16, 'color':'red'})


plt.show()
'''




#tick 能见度
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = 0.1*x

plt.figure()
# 在 plt 2.0.2 或更高的版本中, 设置 zorder 给 plot 在 z 轴方向排序
plt.plot(x, y, linewidth=10, zorder=1)
plt.ylim(-2, 2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

for label in ax.get_xticklabels()+ax.get_yticklabels():
	label.set_fontsize(12)				#调整坐标轴刻度字体大小
	label.set_bbox(dict(facecolor='None', edgecolor='black', alpha=0.8, zorder=10))
	#bbox设置坐标轴刻度的具体风格。facecolor调节box前景色，edgecolor设置边框，alpha设置透明度
plt.show()









