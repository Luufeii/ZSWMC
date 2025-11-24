from torch.utils.data import Dataset 
import torch

class MyDataSet(Dataset):

    def __init__(self, wm_matrix, label):
        super(MyDataSet, self).__init__()
        self.x = wm_matrix
        self.y = label

    def __getitem__(self, index):  # index是必须要有的，在后面访问某个数据的时候，会自动调用这个函数
        # 遍历dataloader时，dataloader其实是从这个函数里取样本，所以这个函数的内容应该尽可能简单，不然会导致dataloader遍历过慢
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)