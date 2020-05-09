# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:26:44 2020

@author: Linda Samsinger

Download Images form Google Image Search using an API 

"""



# to specify 
IMAGE = 'ultramarine'
FOLDER_PATH = r'D:\thesis\images\google\ultramarine'

#%%
### ALL IMAGES IN FOLDER ###

# openly 

## load modules
#from tkinter import *
#from tkinter.filedialog import askdirectory 
#from skimage.io import imread_collection
#
## pop up interface 
#root= Tk()    
#drcty = askdirectory(parent=root,title='Choose directory with image sequence stack files')
#print(drcty)
#path = str(drcty) + '/*.jpg'
#imgs = imread_collection(path) 
#root.destroy()
#
#print(f"Loaded {len(imgs)} images...")

#%%

# silently 

import os

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(FOLDER_PATH):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

for f in files:
    print(f)
    
#%%
# find average color of an image
    
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import statistics as s

# load and show image in BGR 
image = cv2.imread(files[15]) # BGR with numpy.uint8, 0-255 val 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image) #now it is in RGB 
plt.axis('off')
plt.show()
image.shape

# crop image (remove white surroundings)
crop_img = image[80:100,80:100]

# calculate average RGB
average = image.mean(axis=0).mean(axis=0)
average = crop_img.mean(axis=0).mean(axis=0)

# show average color
a = np.full((100, 100, 3), average, dtype=np.uint8)
plt.title("Average Color")
plt.imshow(a) #now it is in RGB 
plt.axis('off')
plt.show()

#%%

# find average color of all images

avgs = []
for f in files: 
    image = cv2.imread(f) # BGR with numpy.uint8, 0-255 val 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try: 
        # crop image 
        image = image[100:image.shape[0]-100,100:image.shape[1]-100]
    except: 
        pass
    # calculate average RGB
    average = crop_img.mean(axis=0).mean(axis=0)
    avgs.append(list(average))

# average of averages 
avgs = np.array(avgs)
avgavg = avgs.mean(axis=0)
avgcolor = list(avgavg)
avgcolor = round(avgcolor[0]), round(avgcolor[1]), round(avgcolor[2])
print(f"Average RGB color across all images: {avgcolor}")

# show average of averages color
a = np.full((100, 100, 3), avgavg, dtype=np.uint8)
plt.imshow(a) #now it is in RGB 
plt.axis('off')
plt.show()



#%%

# make dataframe 

def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

def lab2lch(lab, h_as_degree = False):
    """
    :param lab: np.ndarray (dtype:float32) l : {0, ..., 100}, a : {-128, ..., 128}, l : {-128, ..., 128}
    :return: lch: np.ndarray (dtype:float32), l : {0, ..., 100}, c : {0, ..., 128}, h : {0, ..., 2*PI}
    """
    if not isinstance(lab, np.ndarray):
        lab = np.array(lab, dtype=np.float32)
    lch = np.zeros_like(lab, dtype=np.float32)
    lch[..., 0] = lab[..., 0]
    lch[..., 1] = np.linalg.norm(lab[..., 1:3], axis=len(lab.shape) - 1)
    lch[..., 2] = np.arctan2(lab[..., 2], lab[..., 1])
    lch[..., 2] += np.where(lch[..., 2] < 0., 2 * np.pi, 0)
    if h_as_degree:
        lch[..., 2] = (lch[..., 2] / (2*np.pi)) * 360
    return lch

def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (red, green, blue)


r,g,b = avgcolor
srgb = [r,g,b]

avgcolor2 = r/255,g/255,b/255
lab1 = convert_color(avgcolor2, cv2.COLOR_RGB2Lab)
l,a,b = lab1.tolist()
lab = np.round(l), np.round(a), np.round(b)
lab = list(lab)

hsv = convert_color(avgcolor2, cv2.COLOR_RGB2HSV)
h,s,v = list(hsv)
hsv = np.round(h), np.round(s), np.round(v)
hsv = list(hsv)

lch = lab2lch(lab1)
l,c,h = list(lch)
lch = np.round(l), np.round(c), np.round(h)
lch = list(lch)

r,g,b = avgcolor
hx = rgb_to_hex(int(r), int(g), int(b))

col_decl = pd.DataFrame({'VIAN_color_category': 'ultramarine', 
                         'srgb': [srgb],
                         'cielab': [lab],
                         'hsv': [hsv],
                         'LCH': [lch],
                         'HEX': hx})

FOLDER_PATH = r'D:\thesis\images\google\ultramarine'
os.chdir(FOLDER_PATH)

col_decl.to_csv('ultramarine.csv')