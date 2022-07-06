from torch.utils.data import Dataset
import os
from  PIL import  Image

class Mydata(Dataset):
    #初始化类  全局变量
    def __int__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dit = label_dir
        self.path = os.path.join(self.root_dir+self.label_dit)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dit,img_name)
        img = Image.open()
        label = self.label_dit
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dit = 'demo01/dataset/train'
ants_label_dir = "ants"
ants_dataset = Mydata(root_dit,ants_label_dir)