'''
# Radiobutton 选择按钮
import tkinter as tk

window = tk.Tk()
window.title('My Window')
window.geometry('200x200')

var = tk.StringVar()
l = tk.Label(window, bg='#FFE7BA', width=30, text='empty')
l.pack()

def print_selection():
	l.config(text='You have selected --' + var.get())
	pass

#创建选项
r1 = tk.Radiobutton(window, text='Option A', variable=var, value='A', command=print_selection)
r1.pack()
r2 = tk.Radiobutton(window, text='Option B', variable=var, value='B', command=print_selection)
r2.pack()
r3 = tk.Radiobutton(window, text='Option C', variable=var, value='C', command=print_selection)
r3.pack()

window.mainloop()
'''









'''
# Scale 滑块
import tkinter as tk

window = tk.Tk()
window.title('My Window')
window.geometry('200x200')

l = tk.Label(window, bg='#FFE7BA', width=30, text='empty')
l.pack()

def print_selection(v):
	l.config(text='You have selected ' + v)

# 定义scale
s = tk.Scale(window, label='Adjustment button', from_=0, to=100, orient=tk.HORIZONTAL,
		length=200, showvalue=1, tickinterval=20, resolution=1, command=print_selection)			#orient=tk.HORIZONTAL是指滑块为横向，tickinterval=20为标签单位长度，resolution=1选择数据精度
s.pack()

window.mainloop()
'''











'''
# checkbutton 勾选项
import tkinter as tk

window = tk.Tk()
window.title('My Window')
window.geometry('200x200')

l = tk.Label(window, bg='#FFE7BA', width=30, text='empty')
l.pack()

def print_selection():
	if(var1.get() == 1) & (var2.get() == 0):
		l.config(text='I love Python')
	elif(var1.get() == 0) & (var2.get() == 1):
		l.config(text='I love Java')
	elif(var1.get() == 0) & (var2.get() == 0):
		l.config(text='I do not love either')
	else:
		l.config(text='I love both')

var1 = tk.IntVar()
var2 = tk.IntVar()


# 创建checkbutton
c1 = tk.Checkbutton(window, text='Python', width=8, variable=var1, onvalue=1, offvalue=0, command=print_selection)
c1.pack()
c2 = tk.Checkbutton(window, text='Java', width=8, variable=var2, onvalue=1, offvalue=0, command=print_selection)
c2.pack()

window.mainloop()
'''











'''
# Canvas 画布
import tkinter as tk

window = tk.Tk()
window.title('My Window')
window.geometry('400x400')

# 创建画布
canvas = tk.Canvas(window, bg='#FFE7BA', height=200, width=400)
# 插入图片
# image_file = tk.PhotoImage(file='')
# image = canvas.create_image(100,100, anchor='center', image=image_file)
# 创建多边形
x0, y0, x1, y1 = 50, 50, 80, 80
line = canvas.create_line(x0, y0, x1, y1)
oval = canvas.create_oval(x0, y0, x1, y1, fill='red')
arc = canvas.create_arc(x0+40, y0+40, x1+40, y1+40, start=0, extent=130)
rect = canvas.create_rectangle(x0+100, y0+100, x1+50, y1+50)

canvas.pack()

# 创建移动button
def moveit():
	canvas.move(rect, 0, 2)
	canvas.move(oval, 3, 0)
b = tk.Button(window, text='Move', command=moveit).pack()

window.mainloop()
'''













# Menubar 菜单
import tkinter as tk

window = tk.Tk()
window.title('My Window')
window.geometry('400x400')

l = tk.Label(window, bg='#FFE7BA', width=30, text='')
l.pack()

# 定义menubar
counter = 0
def do_job():
	global counter
	l.config(text='do' + str(counter))
	counter += 1

menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='New', command=do_job)
filemenu.add_command(label='Open', command=do_job)
filemenu.add_command(label='Save', command=do_job)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=window.quit)

editmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='Edit', menu=editmenu)
editmenu.add_command(label='Cut', command=do_job)
editmenu.add_command(label='Copy', command=do_job)
editmenu.add_command(label='Paste', command=do_job)

submenu = tk.Menu(filemenu)
filemenu.add_cascade(label='Import', menu=submenu, underline=0)
submenu.add_command(label='Submenu1', command=do_job)

window.config(menu=menubar)

window.mainloop()