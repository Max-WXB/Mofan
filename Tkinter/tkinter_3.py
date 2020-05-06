'''
# Frame 框架
import tkinter as tk 

window = tk.Tk()
window.title('My window')
window.geometry('400x400')

tk.Label(window, text='on the window').pack()

# 定义主框架 frame
frm = tk.Frame(window)
frm.pack()
frm_l = tk.Frame(frm, )
frm_r = tk.Frame(frm, )
frm_l.pack(side='left')
frm_r.pack(side='right')

tk.Label(frm_l, text='on the frm_l 1').pack()
tk.Label(frm_l, text='on the frm_l 2').pack()
tk.Label(frm_r, text='on the frm_r 1').pack()

window.mainloop()
'''












'''
# messagebox 窗口
import tkinter as tk 
from tkinter import messagebox

window = tk.Tk()
window.title('My window')
window.geometry('400x400')

# 创建弹窗
def hit_me():
	tk.messagebox.showinfo(title='Hi', message='hahahahahah')
	tk.messagebox.showwarning(title='Warning', message='nononononono')
	tk.messagebox.showerror(title='Error', message='hahahahahah')
	print(tk.messagebox.askquestion(title='Hi', message='hahahahahah'))			# return yes or no
	print(tk.messagebox.askyesno(title='Hi', message='hahahahahah'))				#return true or false
	print(tk.messagebox.askokcancel(title='Hi', message='hahahahahah'))
	print(tk.messagebox.askretrycancel(title='Hi', message='hahahahahah'))

tk.Button(window, text='Hit me', command=hit_me).pack()

window.mainloop()
'''














# pack(), grid(), place()放置位置
import tkinter as tk 

window = tk.Tk()
window.title('My window')
window.geometry('400x400')

# pack()
tk.Label(window, text='pack_top').pack(side='top')
tk.Label(window, text='pack_bottom').pack(side='bottom')
tk.Label(window, text='pack_left').pack(side='left')
tk.Label(window, text='pack_right').pack(side='right')

# # grid()
# for i in range(4):
# 	for j in range(3):
# 		tk.Label(window, text='grid').grid(row=i,column=j,ipadx=10,ipady=10)

# place()
tk.Label(window, text='place').place(x=200, y=200, anchor='center')

window.mainloop()