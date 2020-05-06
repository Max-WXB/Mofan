'''
# label 和 button
import tkinter as tk

# 创建Window并命名
window = tk.Tk()
window.title('my window')
window.geometry('600x400')

# 定义全局变量中label中显示的内容
var = tk.StringVar()
# 定义label
l = tk.Label(window, textvariable=var, bg='#FFE7BA', font=('Arial', 20), width=80, height=10)
# 放置label的位置
l.pack()			#放置一个相对位置

on_hit = False

def hit_me():
	global on_hit
	if on_hit == False:
		on_hit = True
		var.set('You hit me !?')
	else:
		on_hit = False
		var.set('Please hit button')

# 创建button
b = tk.Button(window, text='Hit me!', font=('Arial', 20), width=30, height=5, command=hit_me)

b.pack()

window.mainloop()		#window循环刷新
'''








'''
# Entry & Text 输入，文本框
import tkinter as tk

# 创建Window并命名
window = tk.Tk()
window.title('my window')
window.geometry('200x200')

# 定义Entry
e = tk.Entry(window, show='*')
e.pack()

# 定义insert button功能
def insert_point():
	var = e.get()
	t.insert('insert', var)

# 定义insert end功能
def insert_end():
	var = e.get()
	t.insert('end', var)

# 定义insert everywhere功能
def insert_everywhere():
	var = e.get()
	t.insert(1.1, var)			#在第一行第一列进行插入
	pass

# 创建button
b1 = tk.Button(window, text='Insert point', font=('Arial', 15), command=insert_point)
b1.pack()
b2 = tk.Button(window, text='Insert end', font=('Arial', 15), command=insert_end)
b2.pack()
b3 = tk.Button(window, text='Insert everywhere', font=('Arial', 15), command=insert_everywhere)
b3.pack()

t = tk.Text(window, height=5)
t.pack()

window.mainloop()
'''











# Listbox 列表控件
import tkinter as tk

# 创建Window并命名
window = tk.Tk()
window.title('my window')
window.geometry('400x400')

var1 = tk.StringVar()
l = tk.Label(window, bg='#FFE7BA', font=('Arial', 20), width=80, height=3, textvariable=var1)
l.pack()

# 定义insert button功能
def print_selection():
	value = lb.get(lb.curselection())		#传入list_box中选定的值
	var1.set(value)

# 创建button
b = tk.Button(window, text='Print Slection', font=('Arial', 15), command=print_selection)
b.pack()

var2 = tk.StringVar()
var2.set((11,22,33,44))
# 定义list_box
lb = tk.Listbox(window, listvariable=var2)
# 向list_box中插入新的
list_items = [1,2,3,4]
for item in list_items:
	lb.insert('end', item)
# 向list_box中插入新的值
lb.insert(0, '1024')
lb.insert(7, "Hello World!")
# 删除list_box中的值
lb.delete(2)
lb.pack()

window.mainloop()



