import numpy as np
import torch
class DATA():
    def __init__(self):
        data_path = '../DATA/denoise_WM.npz'
        data = np.load(data_path)
        self.original_wm = data['original_wm']
        self.label_one_hot = data['label_one_hot']
        self.denoise_wm = data['denoise_wm']
        self.label_name = data['label_name']
        self.Total_Num = len(data['original_wm'])


    # 下面的代码每次执行会浪费大量时间，所以把降噪后的晶圆图保存下来直接读取
    '''def trans(self,arr):
        dic = {0:'C',1:'D',2:'EL',3:'ER',4:'L',5:'NF',6:'S',7:'R'}
        dic_complete = {0:'Center',1:'Donut',2:'Edge_loc',3:'Edge_ring',4:'Loc',5:'Nearfull',6:'Scratch',7:'Random'}
        # 单一缺陷我们使用完整的缺陷名，与defect_att中的键名对应一致
        str1 = ''
        if arr.sum() == 0:
            str2 = 'Normal'
        else:
            if arr.sum() == 1:
                for i in range(len(arr)):
                    if arr[i] == 1:
                        str2 = dic_complete[i]
                        break
            else:
                for i in range(len(arr)):
                    if arr[i] == 1:
                        str1 = str1+dic[i]+'+'
                    str2 = str1[:-1]  # 去除最后的加号，这个数组切片的作用是去除最后一个元素
        return str2
    
    def denoise(self,wafermap):
        wm = wafermap.copy()  # wm = wafermap，这样的话，对wm的修改操作也会对wafermap进行修改操作
        # wm2 = wafermap.copy()  # 我们用窗口滑动wm，但修改操作用在wm2上，这样能防止因为消除坏晶粒引起来的连锁反应，会导致晶圆图几乎清空
        # 这就是copy()能够只传值，这样对wm操作就不会影响wafermap。
        row_len = len(wm)  # 获取晶圆图的行数，这里是52
        column_len = len(wm[0])  # 获取晶圆图的列数，这里是52
        threshold = 4/9   # 滤波的阈值
        w_size = 9  # 我们选择3x3的滤波窗口
        for no in range(4):  # 一张晶圆图经过4次滤波效果达到最好，所以这里循环4次
            for i in range(1,row_len-1):  # 我们忽略最边角的晶粒，因为边角的晶粒无法获取完整的滤波窗口，需要进行特殊处理，在此直接忽略不计
                for j in range(1,column_len-1):
                    num = 0  # 记录坏晶粒的个数
                    if wm[i][j] == 2:  # 只有坏晶粒需要滤波，背景(0)和好晶粒(1)则不需要
                        for row in range(i-1,i+2):
                            for column in range(j-1,j+2):  # 以(i,j)为中心的三行三列
                                if wm[row][column] == 2:
                                    num = num + 1
                        result = num/w_size
                        if result < threshold:
                            wm[i][j] = 1
        return wm


    def apply_all_func(self):
        self.label_list = []  #  记录每个晶圆图的标签
        self.denoise_wm_list = []
        for i in range(self.Total_Num):
            self.label_list.append(self.trans(self.one_hot[i]))
            self.denoise_wm_list.append(self.denoise(self.matrix[i]))
        self.label = np.array(self.label_list)
        self.denoise_wm = np.array(self.denoise_wm_list)'''
        