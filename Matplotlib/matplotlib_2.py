'''
#scatter 散点图
import matplotlib.pyplot as plt 
import numpy as np 

n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
T = np.arctan2(Y,X)			#随机生成好看的颜色

plt.scatter(X, Y, s=75, c=T, alpha=0.6)		#创建散点图，点由XY定位，s代表size，c代表color，alpha代表透明度
plt.scatter(np.arange(10), np.arange(10))

plt.xlim((-1.5, 1.5))		#限定x轴上点在-1.5~1.5之间
plt.ylim((-1.5, 1.5))		#限定y轴上点在-1.5~1.5之间
plt.xticks(())				#不显示x轴坐标轴
plt.yticks(())				#比现实y轴坐标轴

plt.show()
'''



'''
#bar 柱状图
import matplotlib.pyplot as plt
import numpy as np


n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')		#创建bar图，+代表图显示在双轴bar图的上半部分，facecolor是填充颜色，edgecolor是边框颜色
plt.bar(X, -Y1, facecolor='#ff9999', edgecolor='white')		#- 代表创建的图为双轴bar图的下半部分

for x,y in zip(X, Y1):			#zip是指将X和Y1数值分别传递给x和y
	plt.text(x+0.04, y+0.05, '%.2f'%y, ha='center', va='bottom')
	#x+0.4和y+0.05指标注的位置，%.2f指保留两位小数，ha代表横向居中，va代表纵向底部对齐
for x,y in zip(X, Y2):
	plt.text(x+0.04, -y-0.05, '%.2f'%y, ha='center', va='top')

plt.xlim(-0.5,n)
plt.xticks(())
plt.ylim(-1.25,1.25)
plt.yticks(())

plt.show()
'''




'''
#Contours 等高线图
import matplotlib.pyplot as plt
import numpy as np

def f(x,y):
	#计算高度的函数
	return (1 - x/2 + x**5 + y**3)*np.exp(-x**2 - y**2)

n = 256
x = np.linspace(-3, 3, n)		#取值范围是-3~3，有256个点
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x,y)			#meshgrid在二维平面中将每一个x和y分别对应起来，编制成栅格

plt.contourf(X, Y, f(X,Y), 8, alpha=0.75, cmap=plt.cm.hot)
#创建等高线图，8代表等高线密集程度，alpha是透明度，color map将暖色组分配给不同的数值

C = plt.contour(X, Y, f(X,Y), 8, colors='black', linewidths=0.5)
#创建图中等高线，同样密集程度为8，颜色为黑色，线条粗细为0.5

plt.clabel(C, inline=True, fontsize=10)
#为等高线条件标注，inline为在线内标注，fontsize为字体大小

plt.xticks(())
plt.yticks(())
plt.show()
'''



'''
#Image 图片
import matplotlib.pyplot as plt
import numpy as np

a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')
#创建image，interpolation是显示色块的效果

plt.colorbar(shrink=0.5)
#显示旁边的colorbar，shrink为colorbar的比例大小

plt.xticks(())
plt.yticks(())
plt.show()
'''




#3D 数据
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)			#显示3D坐标轴

X = np.arange(-4, 4, 0.25)		#创建x，y的值
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)		#与等高线一样，创建点格栅
R = np.sqrt(X ** 2 + Y ** 2)	#计算X和Y的值对应的高度值
Z = np.sin(R)					#显示高度值

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), edgecolor='black')
#restride是指row的跨度，cstride是column的跨度

ax.contourf(X, Y, Z, zdir='z',offset=-2, cmap='rainbow')		#创建z轴映射上的等高线图
ax.contourf(X, Y, Z, zdir='x',offset=-4, cmap='rainbow')
ax.set_zlim(-2,2)		#限制等高线图的位置
plt.show()


