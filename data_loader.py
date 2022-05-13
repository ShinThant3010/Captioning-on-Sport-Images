#!/usr/bin/env python
# coding: utf-8

# In[3]:


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import pandas as pd
import torch

class SportDataset(Dataset):
    def __init__(self, image_paths, captionInIdx_file, transform, mode):
        super().__init__()
        self.paths = image_paths
        self.captionInIdx_file = captionInIdx_file
        self.len = len(self.paths)
        self.transform = transform
        self.mode = mode
        
    def __len__(self): return self.len
    
    def __getitem__(self, index): 
        # obtain image and caption
        path = self.paths[index]
        
        # convert image to tensor and pre-process using transform
        PLI_image = Image.open(path).convert('RGB')
        copy_image = PLI_image.copy()
        copy_image = copy_image.resize((100, 100), Image.ANTIALIAS)
        original_image = np.array(copy_image)
        image = self.transform(PLI_image)
        imageName = os.path.basename(path)
        #print(imageName)
        
        if self.mode == 'train':

            ### get csv of captions_idx and images
            df = pd.read_csv(self.captionInIdx_file)
            df.head()

            ### get caption of respective imageName
            respect_cap_str = df.loc[df['img_name'] == imageName, 'Caption'].values[0]
            respect_cap_strList = respect_cap_str.strip('][').split(', ')
            respect_cap = [int(i) for i in respect_cap_strList]

            # convert caption to tensor of word ids
            caption = torch.Tensor(respect_cap).long()

            # return pre-processed image and caption tensors
            return (image, caption)
        
        else:
            return (original_image, image, imageName)


# In[ ]:




