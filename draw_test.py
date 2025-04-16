# Code modified from: https://zhuanlan.zhihu.com/p/138232835

import tkinter
from PIL import Image
import tkinter.simpledialog
import tkinter.colorchooser
import tkinter.filedialog
from PIL import Image, ImageTk, ImageGrab
import time
import numpy as np


def center_window(top, w, h):
    # 获取屏幕 宽、高
    ws = app.winfo_screenwidth()
    hs = app.winfo_screenheight()
    app.geometry('%dx%d' % (w, h))


# 保存画布
def getter(widget):  # 参数是画布的实体
    time.sleep(0.5)  # 等待一会，否则会把点击“保存”那一刻也存进去
    x = app.winfo_x() + widget.winfo_x()  # 不知道为什么会有偏差，所以进行了微调，扩大了截图范围
    y = app.winfo_y() + widget.winfo_y()
    if app.winfo_x() < 0:  # 获取的位置有问题，有可能为负数
        x = 0
    if app.winfo_y() < 0:
        y = 0
    x1 = x + widget.winfo_width() + 200
    y1 = y + widget.winfo_height() + 200
    filename = tkinter.filedialog.asksaveasfilename(filetypes=[('.jpg', 'JPG')],
                                                    initialdir='')
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

def catch_figure(widget):  # 参数是画布的实体
    time.sleep(0.5)  # 等待一会，否则会把点击“保存”那一刻也存进去
    x = app.winfo_x() + widget.winfo_x() # 不知道为什么会有偏差，所以进行了微调，扩大了截图范围
    y = app.winfo_y() + widget.winfo_y()
    if app.winfo_x() < 0:  # 获取的位置有问题，有可能为负数
        x = 0
    if app.winfo_y() < 0:
        y = 0

    x = x + 100
    y = y + 100

    x1 = x + widget.winfo_width() + 200
    y1 = y + widget.winfo_height() + 200

    img = ImageGrab.grab().crop((x, y, x1, y1))

    # 放缩为 28x28大小
    img = img.resize((28, 28))
    # 转为灰度图
    img = img.convert('L')
    random_name = str(time.time()) + '.jpg'
    img.save(f'./figs/{random_name}')
    #导出为npy数组
    nimg = np.array(img)/255.0
    nimg = nimg.reshape(1, 1, 28, 28)

    # 预测
    pred = model(nimg)
    label = np.argmax(pred)
    print(label)


# 鼠标左键单击，允许画图
def onLeftButtonDown(event):
    yesno.set(1)
    X.set(event.x)
    Y.set(event.y)
    if what.get() == 4:
        canvas.create_text(event.x, event.y, font=("微软雅黑", int(size)), text=text, fill=foreColor)
        what.set(1)


# 按住鼠标左键移动，画图
def onLeftButtonMove(event):
    global lastDraw
    if yesno.get() == 0:
        return
    if what.get() == 1:
        # 使用当前选择的前景色绘制曲线
        lastDraw = canvas.create_line(X.get(), Y.get(), event.x, event.y, fill=foreColor,
                                      width=80, capstyle=tkinter.ROUND)  # 返回值就是对图形的计数，直接delete这个数字就能删除该图形
        X.set(event.x)
        Y.set(event.y)
    elif what.get() == 2:
        try:
            canvas.delete(lastDraw)
        except Exception as e:
            pass
        # 绘制直线，先删除刚刚画过的直线，再画一条新的直线
        lastDraw = canvas.create_line(X.get(), Y.get(), event.x, event.y,
                                      fill=foreColor)
    elif what.get() == 3:
        # 绘制矩形，先删除刚刚画过的矩形，再画一个新的矩形
        try:
            canvas.delete(lastDraw)
        except Exception as e:
            pass
        lastDraw = canvas.create_rectangle(X.get(), Y.get(), event.x, event.y,
                                           outline=foreColor)

    elif what.get() == 5:
        # 橡皮，使用背景色填充10*10的矩形区域
        lastDraw = canvas.create_rectangle(event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                                           outline=backColor)
    elif what.get() == 6:
        # 绘制圆形，先删除刚刚画过的矩形，再画一个新的矩形
        try:
            canvas.delete(lastDraw)
        except Exception as e:
            pass
        lastDraw = canvas.create_oval(X.get(), Y.get(), event.x, event.y,
                                      fill=backColor, outline=foreColor)


# 鼠标左键抬起，不允许画图
def onLeftButtonUp(event):
    global lastDraw
    if what.get() == 2:
        # 绘制直线

        lastDraw = canvas.create_line(X.get(), Y.get(), event.x, event.y, fill=foreColor)
    elif what.get() == 3:

        lastDraw = canvas.create_rectangle(X.get(), Y.get(), event.x, event.y, outline=foreColor)
    elif what.get() == 6:

        lastDraw = canvas.create_oval(X.get(), Y.get(), event.x, event.y, outline=foreColor)
    yesno.set(0)
    end.append(lastDraw)


# 鼠标右键抬起，弹出菜单
def onRightButtonUp(event):
    menu.post(event.x_root, event.y_root)


# 打开图像文件
def Open():
    filename = tkinter.filedialog.askopenfilename(title='导入图片',
                                                  filetypes=[('image', '*.jpg *.png *.gif')])
    if filename:
        global image

        image = Image.open(filename)
        image = image.resize((100, 100), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        canvas.create_image(100, 100, image=image)


# 添加菜单，清除
def Clear():
    global lastDraw, end
    for item in canvas.find_all():
        canvas.delete(item)
    end = [0]
    lastDraw = 0


# 撤销
def Back():
    global end
    try:
        for i in range(end[-2], end[-1] + 1):  # 要包含最后一个点，否则无法删除图形
            canvas.delete(i)
        end.pop()  # 弹出末尾元素
    except:
        end = [0]


# 保存
def Save():
    getter(canvas)

def Catch():
    catch_figure(canvas)

def drawCurve():
    what.set(1)

def drawLine():
    what.set(2)

def drawRectangle():
    what.set(3)

def drawCircle():
    what.set(6)

# 文本框
def drawText():
    global text, size
    text = tkinter.simpledialog.askstring(title='输入文本', prompt='')
    if text != None:
        size = tkinter.simpledialog.askinteger('输入字号', prompt='', initialvalue=20)  # 默认值为20
        if size == None:
            size = "20"
    what.set(4)

# 选择前景色
def chooseForeColor():
    global foreColor
    foreColor = tkinter.colorchooser.askcolor()[1]


# 选择背景色
def chooseBackColor():
    global backColor
    backColor = tkinter.colorchooser.askcolor()[1]

# 橡皮
def onErase():
    what.set(5)


if __name__ == '__main__':
    import mynn as nn
    # 导入模型
    model = nn.models.Model_CNN_v2_1()
    model.load_model(r'.\saved_models\model_v2_1_1\best_model.pickle')


    # 创建窗口
    app = tkinter.Tk()
    # 设置窗口位置
    app.geometry('700x700+10+10')

    # 设置窗口标题
    app.title('我的画图')
    # center_window(app, 700, 700)
    # 控制是否允许画图的变量，1：允许，0：不允许
    yesno = tkinter.IntVar(value=0)
    # 控制画图类型的变量，1：曲线，2：直线，3：矩形，4：文本，5：橡皮 6：圆形
    what = tkinter.IntVar(value=1)
    # 记录鼠标位置的变量
    X = tkinter.IntVar(value=0)
    Y = tkinter.IntVar(value=0)
    # 前景色
    foreColor = '#FFFFFF'
    backColor = '#000000'
    # 创建画布
    image = tkinter.PhotoImage()
    canvas = tkinter.Canvas(app, bg='white', width=700, height=700, background=backColor)
    canvas.create_image(100, 100, image=image)
    # 记录最后绘制图形的id
    lastDraw = 0
    end = [0]  # 每次抬起鼠标时，最后一组图形的编号
    size = "20"  # 初始字号

    canvas.bind('<Button-1>', onLeftButtonDown)  # 单击左键
    canvas.bind('<B1-Motion>', onLeftButtonMove)  # 按住并移动左键
    canvas.bind('<ButtonRelease-1>', onLeftButtonUp)  # 释放左键
    canvas.bind('<ButtonRelease-3>', onRightButtonUp)  # 释放右键
    canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)

    # 菜单栏
    menu = tkinter.Menu(app, tearoff=0)
    menu.add_command(label='导入', command=Open)
    menu.add_command(label='清屏', command=Clear)
    menu.add_command(label='撤销', command=Back)
    menu.add_command(label='保存', command=Save)
    menu.add_command(label='识别', command=Catch)
    menu.add_separator()  # 添加分割线
    menuType = tkinter.Menu(menu, tearoff=0)
    menuType.add_command(label='铅笔', command=drawCurve)
    menuType.add_command(label='直线', command=drawLine)
    menuType.add_command(label='矩形', command=drawRectangle)
    menuType.add_command(label='圆形', command=drawCircle)
    menuType.add_command(label='文本', command=drawText)
    menuType.add_separator()
    menuType.add_command(label='选择前景色', command=chooseForeColor)
    menuType.add_command(label='选择背景色', command=chooseBackColor)
    menuType.add_command(label='橡皮擦', command=onErase)

    menu.add_cascade(label='工具栏', menu=menuType)

    app.mainloop()
