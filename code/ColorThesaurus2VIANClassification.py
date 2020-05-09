# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:49:56 2020

@author: Linda Samsinger

Image-to-Label Classification

The goal is to help make a dataset with images and corresponding labels. 
Where an image is given, the user can key in the label when prompted. 

At the end all images have labels. 
"""


# import modules
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

### Color-Thesaurus EPFL ###

PATH = r'D:\thesis\code\pd28vianhues'
FILE = 'SRGBLABhsvhslLCHHEX_Eng_VIANHuesColorThesaurus.xlsx'


# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ", index_col=0)

data[['name','srgb']]

#%%
### Processing ###

    
def plot_color(color, size): 
    # plot image
    image = np.full((size, size, 3), color, dtype=np.uint8) / 255
    plt.imshow(image) 
    plt.axis('off')
    plt.show()
    return image 


#%% 

### add missing VIAN color categories to data set 
subdata = data[data['VIAN_color_category'].notnull() == False]

labels = []
for f in range(len(subdata['name'])): 
    # VIAN color categories
    lst = data['VIAN_color_category'].unique()[1:]    
    print("VIAN colors: ", lst)
    # plot test color 
    plot_color(eval(subdata['srgb'].iloc[f]), 10)
    print(subdata['name'].iloc[f])
    label = input("Which VIAN color category should this color have? ")  
    labels.append(label)

subdata['VIAN_color_category'] = labels
subdata[['name','VIAN_color_category' ]]
#data['VIAN_color_category_all'] = labels

# add subset to whole dataset 
data[data['VIAN_color_category'].notnull() == False] = subdata
data[['name','VIAN_color_category']]
data['VIAN_color_category'].isnull().any()

#%%
FILE = 'SRGBLABhsvhslLCHHEX_EngALL_VIANHuesColorThesaurus.xlsx'

# set directory
os.chdir(PATH)

# save dataframe 
data.to_excel(FILE, index=False)