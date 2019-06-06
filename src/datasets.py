import os

from PIL import Image
import numpy as np

from torch.utils.data import Dataset

class BaselineDataset(Dataset):
    def __init__(self, df, img_path, nclasses, trans=None, tensor_trans=None):        
        self.img_path = img_path
        self.trans = trans
        self.tensor_trans = tensor_trans
        self.img_names = df.image.values
        self.nclasses = nclasses
        
        self.ids = df.attribute_ids.values

    def _read_single(self, img_name):
        img_path = os.path.join(self.img_path, img_name)
        return Image.open(img_path)

    def _oh_labels(self, lbls):
        res = np.zeros(self.nclasses, dtype=np.float32)
        res[lbls] = 1
        
        return res
    
    def __getitem__(self, index):
        img = self._read_single(self.img_names[index])
        
        if self.trans:
            img = self.trans(img)
        img = self.tensor_trans(img)
    
        labels = self._oh_labels(self.ids[index])
        
        return img, labels

    def __len__(self):
        return len(self.img_names)


class MixupDataset(Dataset):
    def __init__(self, df, img_path, nclasses, trans=None, tensor_trans=None, alpha=0.4):       
        self.img_path = img_path
        self.trans = trans
        self.tensor_trans = tensor_trans
        self.img_names = df.image.values
        self.nclasses = nclasses
        self.ids = df.attribute_ids.values
        self.alpha = alpha

    def _read_single(self, img_name):
        img_path = os.path.join(self.img_path, img_name)
        return Image.open(img_path)

    def _oh_labels(self, lbls):
        res = np.zeros(self.nclasses, dtype=np.float32)
        res[lbls] = 1
        
        return res
    
    def __getitem__(self, index):
        mix_idx = index
        while mix_idx == index:
            mix_idx = np.random.randint(len(self.img_names))

        img = self._read_single(self.img_names[index])
        img_mix = self._read_single(self.img_names[mix_idx])
        
        if self.trans:
            img = self.trans(img)
            img_mix = self.trans(img_mix)
                           
        labels = self._oh_labels(self.ids[index])
        labels_mix = self._oh_labels(self.ids[mix_idx])
        
        mix_ratio = np.random.beta(self.alpha, self.alpha)
        
        img = np.array(img, dtype=np.float32)
        img_mix = np.array(img_mix, dtype=np.float32)
        
        img = mix_ratio*img + (1 - mix_ratio)*img_mix
        labels = mix_ratio*labels + (1 - mix_ratio)*labels_mix
        
        img = self.tensor_trans(img)
        
        return img, labels

    def __len__(self):
        return len(self.img_names)