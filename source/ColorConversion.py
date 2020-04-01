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

#############
### COLOR ###   
#############
# Color Value Conversion 

# define colors 
light_white = (244,239,208) # image correct
dark_white = (191,186,159)

light_white = (31,54,55) # image correct
dark_white = (169,54,55)

# RGB to HSV 
#light_white = np.array([[[244,239,208]]], dtype='uint8') # needs three brackets around color otherwise invalid number of channels in input image
def floatify(img_uint8): 
    img_floats = img_uint8.astype(np.float32) 
    img_floats = img_floats / 255 
    return img_floats  # RGB to HSV: first step is to divide each channel with 255 to change the range of color from 0-255 to 0-1

light_white = np.array([[[31,54,55]]], dtype=np.uint8)
dark_white = np.uint8([[[230,224,192]]])
light_white = floatify(light_white)

hsv_light_white = cv2.cvtColor(light_white, cv2.COLOR_RGB2HSV)
print(hsv_light_white)
# [[[ 51.666622    0.14754096  0.95686275]]] +- hue value 20 upper limit [[[ 70.666622    0.14754096  0.95686275]]] lower limit [[[ 31.666622    0.14754096  0.95686275]]] 



# HSV to RGB   #TODO: conversion does not work  
def intify(img_f): 
    img_f = img_f * 255
    img_f = np.round(img_f)
    im_ints = img_f.astype(np.uint8)
    return im_ints 

# define colors 
hsv_light_white = np.array([[[51.666622  ,  0.14754096,  0.95686275]]], dtype=np.float32)
# hsv_dark_white = cv2.cvtColor(dark_white, cv2.COLOR_RGB2HSV)
# hsv_light_white = intify(hsv_light_white) 
rgb_light_white = cv2.cvtColor(hsv_light_white, cv2.COLOR_HSV2RGB)
print(rgb_light_white)
# upper : [[[225 244 208]]]; lower : [[[244 215 208]]] 


#%%
# TODO: HSV to RGB 
# RGB to HSV *** HSV to RGB  
def convert_rgbhsv(color, source, dest): 
    """  Convert one specific color 
    - from RGB to HSV 
    - from HSV to RGB
    Arguments: 
        color -- list, original color 
        source -- str, original color space
        dest -- str, target color space
    Returns:
        color -- list, target color """
   
    if source == "RGB" and dest == "HSV": 
        color = np.asarray([[color]], dtype='uint8') # from list to np array of uint8 
        color = color.astype(np.float32)  / 255  # from uint8 to float to 0-1 scale (preprocess for HSV compatibility) 
        color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV) # data input needs to be in image shape with three brackets
        h,s,v = color.tolist()[0][0]
        h = int(round(h,0))
        s = int(round(s,0))
        v = int(round(v,0))
        color = h,s,v 
    elif source == "HSV" and dest == "RGB": 
        pass 
        #print(cv2.cvtColor(color, cv2.COLOR_RGB2HSV))
        #print(cv2.cvtColor(color, cv2.COLOR_HSV2RGB))
    return color 

light_white = (244,239,208)
convert_rgbhsv(light_white, 'RGB', 'HSV')

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
# TODO: debug 
# RGB to L*ab *** L*ab to RGB  
def convert_rgblab(color, source, dest): 
    """  Convert one specific color 
    - from RGB to HSV 
    - from HSV to RGB
    Arguments: 
        color -- list, original color 
        source -- str, original color space
        dest -- str, target color space
    Returns:
        color -- list, target color """
   
    if source == "RGB" and dest == "LAB": 
        r,g,b = color 
      #  if needs CIE XYZ vals first : 
     #  pass 
        
    elif source == "LAB" and dest == "RGB": 
        l,a,b = color
        var_Y = ( l + 16 ) / 116
        var_X = a / 500 + var_Y
        var_Z = var_Y - b / 200
        if var_Y**3  > 0.008856:
            var_Y = var_Y**3
        else: 
            var_Y = ( var_Y - 16 / 116 ) / 7.787
        if var_X**3  > 0.008856: 
            var_X = var_X**3 
        else: 
            var_X = ( var_X - 16 / 116 ) / 7.787
        if var_Z**3  > 0.008856:
            var_Z = var_Z**3
        else: 
            var_Z = ( var_Z - 16 / 116 ) / 7.787
        # Reference-X, Y and Z refer to specific illuminants and observers   
        # Observer= 2°, Illuminant= D65
        X = 95.047 * var_X #* Reference-X=  95.047  
        Y = 100.000 * var_Y #* Reference-Y= 100.000
        Z = 108.883 * var_Z #* Reference-Z= 108.883 
 
        var_X = X / 100
        var_Y = Y / 100
        var_Z = Z / 100
        
        var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986
        var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415
        var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570
        if var_R > 0.0031308:
            var_R = 1.055 * var_R ** ( 1 / 2.4 ) - 0.055;
        else:
            var_R = 12.92 * var_R;
        if var_G > 0.0031308:
            var_G = 1.055 * var_G ** ( 1 / 2.4 ) - 0.055;
        else:
            var_G = 12.92 * var_G;
        if var_B > 0.0031308:
            var_B = 1.055 * var_B ** ( 1 / 2.4 ) - 0.055;
        else:
            var_B = 12.92 * var_B;

        R = var_R #max(int(round(var_R,0)) * 255, 0)
        G = var_G #max(int(round(var_G,0)) * 255, 0)
        B = var_B #int(round(var_B * 255,0))
        color = R, G, B

    return color 

color = (50, -50, 87)
#color = (50, 100, 0)
convert_rgblab(color, 'LAB', 'RGB')[0]
#rgb(255, 0, 123) # correct 

# correct rgbs 
#(250,0,123)
#(247,0,43)
#(204,75,0)
#(140,117,0)
#(31,138,0)
#(0,148,0)
#(0,151,116)
#(0,151,203)
#(0,145,255)
#(0,127,255)
#(123,88,255)
#(220,0,207)




#%%

# HSV: LAB to LCH *** LCH to LAB 
import math 
def convert_lablch(color, source, dest): 
    # converter: ccc.orgfree.com/ # derived formula
    """  Convert one specific color 
    - from CIE-L*ab to CIE-L*ch 
    - from CIE-L*ch to CIE-L*ab
    - CIE-L*ab => luminence =[0,100] from black to white, a = [-100, +100] or [-128, +128] from green to red, b = [-100, +100] or [-128, +128] from blue to yellow 
    - CIE-L*ch => luminence =[0,100] from black to white, c = [0, +100] from green to red, h = [0, 360] from blue to yellow 
    Arguments: 
        color -- list, original color 
        source -- str, original color space
        dest -- str, target color space
    Returns:
        color -- list, target color """
   
    if source == "LAB" and dest == "LCH": 
        l,a,b = color #
        c = int(round(math.sqrt(a**2 + b**2), 0)) #chroma
        if math.degrees(math.atan2(b,a)) >= 0: 
            h = int(round(math.degrees(math.atan2(b,a)),0)) #hue 
        else: 
            h = int(round(math.degrees(math.atan2(b,a)) + 360 ,0)) 
        color = l,c,h 
    elif source == "LCH" and dest == "LAB": 
        l,c,h = color
        a = int(round(c* math.cos((h*math.pi/180)),0))
        b = int(round(c* math.sin((h*math.pi/180)),0))
        color = l,a,b      
    return color 

# color = (100,100,100) # luminence in lch is not 100 on colormine.org although lab has luminence 100 
# convert_lablch(color, "LAB", "LCH")
#(50, 71, 45) #correct 
#(100, 141, 45) #correct 

color = (50,100,0)
convert_lablch(color, "LCH", "LAB")
# color = (50,32,38) #correct 
# color = (20, 39, 7) #correct 


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
