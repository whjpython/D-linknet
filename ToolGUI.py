"""
author:jack
time:2020-12-29
address:huiji-zhengzhou
生成深度学习数据集工具
"""
from tkinter import *
from tkinter.filedialog import askdirectory
import time
from data import change_train_dataset
from tkinter import ttk
from tkinter import messagebox
import threading
import os
root = Tk()
sw = root.winfo_screenwidth()#
sh = root.winfo_screenheight()
ww = 400
wh = 260
x = (sw-ww) / 2
y = (sh-wh) / 2
root.geometry("%dx%d+%d+%d" %(ww,wh,x,y))#居中显示代码
root.title('数据集处理器')
root.resizable(width=False, height=False)#长宽不能变

pic_path = StringVar()#内部定义的字符串变量类型
mask_path = StringVar()#标签的路径
change = StringVar()
out_path = StringVar()
num_class = IntVar()
sign = StringVar()
netchoose = BooleanVar()


#按钮执行命令函数，获取文件名称
def selectPPath():
    #path_ = askopenfilename()
    path_ = askdirectory()
    pic_path.set(path_)
#按钮执行命令函数，获取多个文件名
def selectMPath():
    #path_ = askopenfilenames()
    path_ = askdirectory()
    mask_path.set(path_)
#
def selectOPath():
    path_ = askdirectory()
    out_path.set(path_)


Label(root,text = "影像路径:",fg='Green').grid(row = 0, column = 0)#标签
A = Entry(root, textvariable = pic_path)#文本框
A.grid(row = 0, column = 1)
Button(root, text = "选择影像", fg='blue',command = selectPPath,width=10).grid(row = 0, column = 2,padx=5,pady=5)

Label(root,text = "标签路径:",fg='Green').grid(row = 1, column = 0)
B = Entry(root, textvariable = mask_path)
B.grid(row = 1, column = 1)
Button(root, text = "选择标签", fg='blue', command = selectMPath,width=10).grid(row = 1, column = 2,padx=5,pady=5)

# C = Label(root, textvariable=change,fg = 'red').grid(row=7, column=1,padx = 15,pady=50)#textvariable代替text
Label(root,text="删除字节大小:",fg='Green').grid(row=2,column=0,padx=5,pady=5)
Label(root,text="影像标签格式:",fg='Green',width=10).place(x=192,y=85)
s1=ttk.Combobox(root,values=['tif','png','jpg'],width=5)
s1.place(x=273,y=85)
s1.set('tif')
s2=ttk.Combobox(root,values=['tif','png','jpg'],width=5)
s2.place(x=335,y=85)
s2.set('tif')
F = Entry(root,textvariable = out_path,width=25,fg='blue')#文本框
out_path.set(1000)
F.place(x=140,y=120)
# F.grid(row = 2, column = 1)
#Button(root,text="选择输出", fg='blue',command = selectOPath,width=10).grid(row=2,column = 2,padx=5,pady=5)


Label(root,text = "每张图片分割数量:",fg='Green').grid(row = 4, column = 0,padx=5,pady=5)
D = Spinbox(root,from_=1, to=100, increment=1,textvariable=num_class,width=5).place(x = 140,y=85)


C = Label(root, textvariable=change,fg = 'red').grid(row=6, column=1,padx = 15,pady=20)#textvariable代替text
AAA=False


def Mark():

    while True:
        root.update()
        a=''
        for x in range(6):
            a+='->'
            time.sleep(0.25)
            sign.set(a)


def get_value():

    picpath = A.get()#影像路径
    maskinit = B.get()#标签路径
    numpic = F.get()
    pictype= s1.get()
    masktype=s2.get()
    delnum= num_class.get()

    if not picpath or not maskinit:
        change.set('请选择文件')
        messagebox.showinfo('提示','请选择文件')
    if picpath and maskinit:
        change.set('执行中...！')
        root.update()
        change_train_dataset(picpath,maskinit,delnum,pictype,masktype,int(numpic))
        change.set('完成！')
Button(root, text = "执行", width=10,height=2,command =get_value ,fg='Tomato').grid(row =6, column = 0,padx=30,pady=5)
Button(root,text='退出',width=10,height=2,command=root.quit,fg='Tomato').grid(row=6,column=2,padx = 5,pady=5)


root.mainloop()