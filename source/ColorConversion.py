# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:42:03 2020

@author: lsamsi
"""

# websites 
# converter/match: http://www.easyrgb.com
# converter: convertingcolors.com/
# converter: colormine.org/convert/lab-to-lch 
# brain-teaser: sensing.konicaminolta.us/blog/identfiying-color-differences-using-l-a-b-or-l-c-h-coordinates
# brain-teaser: zschuessler.github.io/DeltaE/learn 


# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import cv2

# color picker: https://www.ginifab.com/feeds/pms/color_picker_from_image.php

# set directory 
os.getcwd()
os.chdir(r'D:\thesis\images')

# load color spaces
# all the color space conversions OpenCV provides 
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# [ 'COLOR_BGR2BGR555','COLOR_BGR2BGR565']
len(flags)
# 274
# The first characters after COLOR_ indicate the origin color space, and 
# the characters after the 2 are the target color space.
flags[40]
# 'COLOR_BGR2HLS'

#############
### IMAGE ###
#############
# load BGR image 
nemo = cv2.imread('nemo.png') # BGR with numpy.uint8, 0-255 val 
print(nemo.shape) 
# (382, 235, 3)
# plt.imshow(nemo)
# plt.show()

# it looks like the blue and red channels have been mixed up. 
# In fact, OpenCV by default reads images in BGR format.
# OpenCV stores RGB values inverting R and B channels, i.e. BGR, thus BGR to RGB: 

# BGR to RGB 
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
plt.imshow(nemo) #now it is in RGB 
plt.show()

#plot a color in image
# color = nemo[381,234]
# print(color)
#[203 151 103] # correct color

# RGB to HSV 
def floatify(img_uint8): 
    img_floats = img_uint8.astype(np.float32) / 255 
    return img_floats

nemo = floatify(nemo)
# convert numpy.uint8 to numpy.float32 after imread for color models (HSV) in range 0-1 requiring floats

# convert RGB to HSV
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV) #last rendering of nemo is in RGB 
plt.imshow(hsv_nemo) #now it is in HSV 
plt.show()
#%%

#############
### COLOR ###   
#############
# Color Value Conversion 

# load modules 
from colormath.color_objects import LabColor, LCHabColor, LCHuvColor, LuvColor, xyYColor, XYZColor, HSLColor, HSVColor, sRGBColor, CMYKColor
from colormath.color_conversions import convert_color

# control: https://www.nixsensor.com/free-color-converter/


# RGB-HSV 
def rgb2hsv(r, g, b): 
    rgb = sRGBColor(r, g, b, is_upscaled=True)
    hsv = convert_color(rgb, HSVColor)
    return hsv 

rgb2hsv(255, 0, 0)
# lab(0, 1, 1) # correct


def hsv2rgb(h, s, v): 
    hsv = HSVColor(h, s, v)
    rgb = convert_color(hsv, sRGBColor)
    rgb = rgb.get_upscaled_value_tuple()
    return rgb 

hsv2rgb(50, .20, .80)
# rgb(204, 197, 163) #correct


# RGB-LAB 
def rgb2lab(r, g, b): 
    rgb = sRGBColor(r, g, b, is_upscaled=True)
    lab = convert_color(rgb, LabColor)
    return lab 

rgb2lab(204, 197, 163)
# lab(79, -3.1, 17.96) # correct

def lab2rgb(l, a, b): 
    lab = LabColor(l, a, b)
    rgb = convert_color(lab, sRGBColor)
    rgb = rgb.get_upscaled_value_tuple()
    return rgb 

lab2rgb(50, -50, 20)
# rgb(0, 139, 82) # correct 

# HSV-LAB 
def hsv2lab(h, s, v): 
    hsv = HSVColor(h, s, v)
    lab = convert_color(hsv, LabColor)
    return lab 

hsv2lab(50, .20, .80)
# lab(79, -3.1, 17.96) # correct

def lab2hsv(l, a, b): 
    lab = LabColor(l, a, b)
    hsv = convert_color(lab, HSVColor)
    return hsv 

lab2hsv(50, -50, 20)
# rgb(155, 1, .5)  # correct


# LAB-LCH  
def lab2lch(l, a, b): 
    lab = LabColor(l, a, b)
    lch = convert_color(lab, LCHabColor)
    return lch 

lab2lch(255, 0, 0)
# lab(53, 80, 67) 

def lch2lab(l, c, h): 
    lch = LCHabColor(l, c, h)
    lab = convert_color(lch, LabColor)
    return lab 

lch2lab(50, -50, 20)
# rgb(0, 139, 82) 




#%%

# HSV: 8-bit to 32-bit *** 32-bit to 8-bit  
def convert_hsv8_32bit(color, source, dest):
    """  Convert one specific HSV color 
    - from 8-bit image HSV to 32-bit image HSV 
    - from 32-bit image HSV to 8-bit image RGB
    - 8-bit image: 0-180°, 0-255, 0-255 scales
    - 32-bit image: 0-360°,0-100,0-100 scales
    Arguments: 
        color -- list, original HSV color 
        source -- str, original HSV bit image space
        dest -- str, target HSV bit image space
    Returns:
        color -- list, target HSV color """
   
    if source == "8-bit" and dest == "32-bit": 
        h,s,v  = color
        assert h <= 180 and s <= 255 and v <= 255 
        h = int(h*2)
        s = int(round( s/255*100, 0))
        v = int(round( v/255*100, 0))
        color = (h, s, v) # to 0-360, 0-100, 0-100
    elif source == "32-bit" and dest == "8-bit": 
        h,s,v  = color
        assert h <= 360 and s <= 100 and v <= 100
        h = int(h/2)
        s = int(round( s/100*255, 0))
        v = int(round( v/100*255, 0))
        color = (h, s, v)  # to 0-180, 0-255, 0-255
    return color 

# 8-bit image in HSV space
dark_orange = (1,190,200)
convert_hsv8_32bit(dark_orange, '8-bit', '32-bit')
# (2, 75, 78)

dark_white = (28,25,82) 
convert_hsv8_32bit(dark_orange, '32-bit', '8-bit')
# (14, 64, 209)








#%%


########################################
### Color Wheel Colors in Dataframes ###
########################################

# Requirement: Color Conversion (see section above)
# Color Wheel 30°-steps in different color spaces 
import pandas as pd 

# HSV-Color Wheel in 30°-steps 
lst1 = [(255,0,0) ,(255,128,0) ,(255,255,0),(128,255,0),(0,255,0),(0,255,128),(0,255,255),(0,128, 255),(0, 0, 255),(128, 0, 255),(255, 0, 255),(255, 0, 128)]
lst2 = []
lst3 = ['red', 'orange', 'yellow', 'green-yellow', 'green', 'green-blue', 'cyan', 'blue-yellow','blue','purple','magenta','red-yellow']

for i in range(len(lst1)): 
    lst2.append(convert_rgbhsv(lst1[i], 'RGB', 'HSV'))
    
df = pd.DataFrame()
df['RGB'] = lst1
df['HSV'] = lst2
df['name'] = lst3

os.chdir(r'D:\thesis\code\pd12hues')
df.to_csv('rgbhsv_12.csv')

#%%

# LCH-Color Wheel in 30°-steps 
# get 12 hues of 30°-steps for L*CH's H-channel 
lablch_twelve = dict()
for i in np.linspace(0,360,num=12, endpoint=False): 
    lch_colors = (50,100,i)
    lab_colors = convert_lablch(lch_colors, "LCH", "LAB")
    lablch_twelve[lch_colors] = lab_colors

# build pd frame with lab lch for 12 30°-step hues


lst = []
lst2 = []
for i in lablch_twelve.items(): 
    # {'LCH': i[0]}
    lst.append(i[0]) 
    lst2.append(i[1])
 
lst3 = []
for i in range(len(lst2)): 
    lst3.append(convert_rgblab(lst2[i], 'LAB', 'RGB'))

lst4 = ['fuchsia', 'red', 'terracotta','olive', 'kelly','leaf', 'teal','atoll', 'azure','blue','purple','lilac']

import pandas as pd  
lablch_12 = pd.DataFrame()
lablch_12['LCH'] =lst
lablch_12['Lab'] =lst2
lablch_12['RGB'] = lst3
lablch_12['name'] = lst4

lablch_12= pd.read_csv('lablchrgb_12_handcorrected.csv')

# save dataframe with 12 lab-lch hues 
os.chdir(r'D:\thesis\code\pd12hues')
lablch_12.to_csv('lablchrgb_12.csv')
lablch_12.to_csv('lablchrgb_12_handcorrected.csv')


########################################
### Color Categories in Color Space ###
########################################

# define epicentre of 6 color categories in color space

# basic 6 colors (by Itten p. 22)
basic_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
basic_colors_rgb  = [(255, 0, 0), (255, 128, 0), (255, 255, 0),  (0, 255, 0),
       (0, 0, 255), (128, 0, 255)]
basic_colors_hsv  = []
basic_colors_lch  = []

for i in range(len(basic_colors_rgb)):
    basic_colors_hsv.append(convert_rgbhsv(basic_colors_rgb[i], 'RGB', 'HSV'))

df = pd.DataFrame()
df['RGB'] = lst1
df['HSV'] = lst2
df['name'] = lst3

os.chdir(r'D:\thesis\code\pd12hues')
df.to_csv('rgbhsv_12.csv')
