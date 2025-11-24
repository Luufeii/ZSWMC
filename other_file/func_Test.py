# 测试时会用到的一些函数
import torch
import random
import numpy as np
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

class Test_Func():
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0') # 只有一张显卡的话，'cuda'和'cuda:0'是一样的
        else:
            self.device = torch.device('cpu')

    def get_data(self,wm,att):   # 从wm和att中随机取一个对象
        ra = random.randint(0,len(wm))
        return wm[ra],att[ra]
    
    def cosine_similarity(self,v1, v2):  # 参数v1,v2是np.array,不能是tensor，可以用np.array()将tensor转换为array
        # 计算两个向量的点积
        dot_product = np.dot(v1, v2)
        # 计算两个向量的模
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # 计算余弦相似度
        similarity = dot_product / (norm_v1 * norm_v2)
        return similarity

    def euclidean_distance(self,v1, v2):  # 参数v1,v2是np.array
        # 计算两个向量之间的欧氏距离
        distance = np.sqrt(np.sum((v1 - v2) ** 2))
        return distance
    
    # 从缺陷属性字典defe_dict中找到与att_tensor距离最近的缺陷类别
    # 使用改进后的校准堆叠
    def min_distance(self,att_tensor, defe_dict, cali = 0, euc_cali = 20):  # 使用欧式距离
        # cali = 1表示使用校准堆叠，校准堆叠只有广义ZSL时使用
        min = math.inf
        att_array = np.array(att_tensor.cpu())  # 将属性张量转换为属性数组,np.array不会修改原始数据
        for label,attribute in defe_dict.items():
            attribute_array = np.array(attribute)  # np.array不会修改原始数据
            dis = self.euclidean_distance(att_array,attribute_array)
            if cali == 1:
                if '+' not in label:
                    dis = dis + euc_cali
            if dis<min:
                min = dis
                rel_label = label
        return rel_label  # 这是一个str类型

    def max_similarity(self,att_tensor, defe_dict, cali = 0, cos_cali = 0.4):  # 使用余弦相似度
        max = -2
        att_array = np.array(att_tensor.cpu())  # 在GPU上的张量无法转换为array，只有在CPU上的张量才能转换为array
        for label,attribute in defe_dict.items():
            attribute_array = np.array(attribute)
            simi = self.cosine_similarity(att_array,attribute_array)
            if cali == 1:
                if '+' not in label:
                    simi = simi - cos_cali
            if simi>max:
                max = simi
                rel_label = label
        return rel_label  # 这是一个str类型
    
    # 计算一个批次(64或者32)中预测正确的个数
    def predict_num(self, outputs, rel_att, defe_att_dic, typ='cos', cali = 0, align = 0.4):
        # defe_att_dic 缺陷类名:属性张量 的字典，根据不同的测试类型选择不同的字典
        rel_labels = []
        outputs_labels = []   # tensor只能将元素为数字类型的列表转换为tensor
        for ra in rel_att:  # 获取样本的真标签
            for La,At in defe_att_dic.items():
                At = At.to(self.device)   # torch.equal()需要两个tensor在两个相同的设备上，例如都在GPU上
                if torch.equal(ra,At):   # 判断张量是否相等要用这个函数，如果使用==，则返回的结果是张量中每个元素是否相等(类型也是一个张量，元素是True或者False)
                    rel_labels.append(La)
                    break
        if typ == 'cos':
            for out in outputs:
                outputs_labels.append(self.max_similarity(out,defe_att_dic,cali,align))
        else:
            for out in outputs:
                outputs_labels.append(self.min_distance(out,defe_att_dic,cali,align))

        right = 0
        for i in range(len(rel_labels)):
            if outputs_labels[i] == rel_labels[i]:
                right = right+1
        return right
    
    def show_result(self,model,wm,att,att_dic):
        wafermap,attribute = self.get_data(wm,att)

        for La,At in att_dic.items():
            if torch.equal(attribute,At):   # 判断张量是否相等要用这个函数，如果使用==，则返回的结果是张量中每个元素是否相等(类型也是一个张量，元素是True或者False)
                label_rel = La
                break

        colors = ['white', 'green', 'red']  # 对应的是0(背景)，1(好晶粒)，2(坏晶粒)
        cmap1 = mcolors.ListedColormap(colors)
        plt.imshow(torch.reshape(wafermap,(224,224)),cmap=cmap1)  # plt也可以直接展示tensor,只要把维度变成二维
        plt.show()

        wafermap =torch.reshape(wafermap,(1,1,224,224))  # 加上一个批次通道
        wafermap = wafermap.to(self.device)
        model.eval()
        with torch.no_grad():
            output = model(wafermap)
        print(f'预测的属性向量{output[0]}')
        print(f'真实的属性向量{attribute}')

        label_euc = self.min_distance(output[0],att_dic)  # 欧式距离计算的标签
        label_cos = self.max_similarity(output[0],att_dic)  # 余弦相似度计算的标签

        print(f'真实标签为：{label_rel}')
        print(f'欧式距离计算的标签为：{label_euc}')
        print(f'余弦相似度计算的标签为：{label_cos}')

    def get_acc(self, model, dataloader, att_dic, data_size, typ='cos', cali = 0, align = 0.3):  # att_dic是对应于dataloader中缺陷类别的缺陷属性字典,data_size是数据的总个数
        total_right_num = 0  # 记录正确的总个数
        with torch.no_grad():   # 这里要进行验证，不需要修改参数，所以不计算梯度
            for data in dataloader:  
                imgs,labels = data

                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(imgs)

                # 计算准确率
                right_num = self.predict_num(outputs,labels, att_dic, typ, cali, align)
                total_right_num = total_right_num + right_num
        acc = total_right_num/data_size
        return acc

    # 展示混淆矩阵
    def show_cm(self, true_label, predict_label, label_order,typ = 'per', size = 10):
        # 生成混淆矩阵
        cm = confusion_matrix(true_label, predict_label, labels=label_order)  # 按照label_order中的顺序给数据排序
        # 将混淆矩阵中的数字转换为百分比
        # 通过将cm中的每个元素除以其所在行的总和（即该类的真实样本数），再乘以100，将其转换为百分比。
        cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100

        plt.figure(figsize=(size,size))  # 这个要定义在sns.heatmap()前面才可以定义大小
        # 使用seaborn来可视化混淆矩阵
        if typ == 'per':
            # fmt=".1f"参数设置注释的格式为保留一位小数的浮点数, cmap="Blues"指定了颜色映射
            sns.heatmap(pd.DataFrame(cm_percentage, index=label_order, columns=label_order), annot=True, fmt=".1f", cmap="Blues")
            plt.title('Confusion Matrix in Percentage')
        else:
            # fmt="d"表示格式化为整数，
            sns.heatmap(pd.DataFrame(cm, index=label_order, columns=label_order), annot=True, fmt="d", cmap="Blues")
            plt.title('Confusion Matrix in Number')
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # 显示图形
        plt.show()

    def show_cm2(self, true_label, predict_label, label_order, typ='per', size=10):
    # 生成混淆矩阵
        cm = confusion_matrix(true_label, predict_label, labels=label_order)
        cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100

        plt.figure(figsize=(size, size))
        
        # 设置全局字体样式
        plt.rcParams.update({
            'font.weight': 'bold',      # 全局字体加粗
            'axes.labelweight': 'bold', # 坐标轴标签加粗
            'axes.titlesize': 16,       # 标题字体大小
            'axes.labelsize': 14,      # 坐标轴标签字体大小
            'xtick.labelsize': 12,      # x轴刻度标签大小
            'ytick.labelsize': 12,      # y轴刻度标签大小
        })
        
        if typ == 'per':
            sns.heatmap(pd.DataFrame(cm_percentage, index=label_order, columns=label_order), 
                    annot=True, fmt=".1f", cmap="Blues",
                    annot_kws={"weight": "bold", "size": 14})  # 数字加粗并增大
            plt.title('Confusion Matrix in Percentage', fontweight='bold')  # 中文标题
        else:
            sns.heatmap(pd.DataFrame(cm, index=label_order, columns=label_order), 
                    annot=True, fmt="d", cmap="Blues",
                    annot_kws={"weight": "bold", "size": 14})  # 数字加粗并增大
            plt.title('Confusion Matrix in Number', fontweight='bold')  # 中文标题
        
        plt.xlabel('Predicted', fontweight='bold')
        plt.ylabel('True', fontweight='bold')
        
        # 获取当前坐标轴并调整刻度标签
        ax = plt.gca()
        ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
        
        plt.show()

    # 获取真实标签
    def predict_label(self, outputs, rel_att, defe_att_dic, typ='cos', cali = 0, align = 0.4):
        # defe_att_dic 缺陷类名:属性张量 的字典，根据不同的测试类型选择不同的字典
        rel_labels = []
        outputs_labels = []   # tensor只能将元素为数字类型的列表转换为tensor
        for ra in rel_att:  # 获取样本的真标签
            for La,At in defe_att_dic.items():
                At = At.to(self.device)   # torch.equal()需要两个tensor在两个相同的设备上，例如都在GPU上
                if torch.equal(ra,At):   # 判断张量是否相等要用这个函数，如果使用==，则返回的结果是张量中每个元素是否相等(类型也是一个张量，元素是True或者False)
                    rel_labels.append(La)
                    break
        if typ == 'cos':
            for out in outputs:
                outputs_labels.append(self.max_similarity(out,defe_att_dic,cali,align))
        else:
            for out in outputs:
                outputs_labels.append(self.min_distance(out,defe_att_dic,cali,align))
        return rel_labels,outputs_labels

    def get_label(self, model, dataloader, att_dic, typ='cos', cali = 0, align = 0.4):
        label_true = []
        label_pre = []
        with torch.no_grad(): 
            for data in dataloader:  
                imgs,labels = data

                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(imgs)

                y_true,y_pre = self.predict_label(outputs,labels, att_dic, typ, cali, align)
                
                label_true = label_true + y_true
                label_pre = label_pre + y_pre

        return label_true,label_pre