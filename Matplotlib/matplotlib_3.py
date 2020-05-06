'''
#Subplot 多合一显示_1
import matplotlib.pyplot as plt

plt.figure()			#创建画布

plt.subplot(2,2,1)			#subplot(2,2,1)将画布分成两行两列,显示在第一个位置
plt.plot([0,1], [0,1])

plt.subplot(2,2,2)			#subplot(2,2,2)将画布分成两行两列,显示在第二个位置
plt.plot([0,1], [0,1])

plt.subplot(2,2,3)			#subplot(2,2,3)将画布分成两行两列,显示在第三个位置
plt.plot([0,1], [0,1])

plt.subplot(2,2,4)			#subplot(2,2,4)将画布分成两行两列,显示在第四个位置
plt.plot([0,1], [0,1])

plt.show()
'''


'''
#Subplot 多合一显示_2
import matplotlib.pyplot as plt

plt.figure()			#创建画布

plt.subplot(2,1,1)			#第一个图占据画布的三个格子
plt.plot([0,1], [0,1])

plt.subplot(2,3,4)			#因为一行有3格，所以第二行第一个为4
plt.plot([0,1], [0,1])

plt.subplot(2,3,5)
plt.plot([0,1], [0,1])

plt.subplot(2,3,6)
plt.plot([0,1], [0,1])

plt.show()
'''


'''
#Subplot 分格显示
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#方法1 subplot2gride
plt.figure()
ax1 = plt.subplot2grid((3,3), (0,0), colspan=3, rowspan=1)		#创建画布为三行三列，第一个图占据三列一行
ax1.plot([1,2],[1,2])
ax1.set_title('ax1_title')
ax1 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=1)
ax1 = plt.subplot2grid((3,3), (1,2), colspan=1, rowspan=2)
ax1 = plt.subplot2grid((3,3), (2,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((3,3), (2,1), colspan=1, rowspan=1)

#方法2 gridspec
plt.figure()
gs = gridspec.GridSpec(3,3)
ax1 = plt.subplot(gs[0,:])
ax2 = plt.subplot(gs[1,:2])
ax3 = plt.subplot(gs[1:,2])
ax4 = plt.subplot(gs[-1,0])
ax5 = plt.subplot(gs[-1,-2])

#方法3 easy to define structure
f,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2, sharex=True, sharey=True)
ax11.scatter([1,2],[1,2])

plt.tight_layout()			#紧凑显示图像
plt.show()
'''


'''
#图中图
import matplotlib.pyplot as plt

fig = plt.figure()

x = [1,2,3,4,5,6,7]
y = [1,3,4,2,5,8,6]

left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')

left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(x, y, 'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('title-in1')

plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(y[::-1], x, 'g')			#y[::-1]为转置图表
plt.xlabel('x')
plt.ylabel('y')
plt.title('title-in2')

plt.show()
'''




#次坐标轴
import matplotlib.pyplot as plt
import numpy as np 

x = np.arange(0, 10, 0.1)
y1 = 0.05*x**2
y2 = -1*y1

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()			#创建双轴
ax1.plot(x, y1, 'g--')
ax2.plot(x, y2, 'b-')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1', color='g')
ax2.set_ylabel('Y2', color='b')

plt.show()











