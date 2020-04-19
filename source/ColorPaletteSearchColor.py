# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:40:15 2020

@author: Linda Samsinger

=====================
Color Palettes with Same Color 
=====================

For a given color, find all color palettes with the same color in them. 
Filter out color palettes which contain the same color. 
"""


########### ColorPalette Extractor ###########

# import modules
import os
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pandas as pd


# to specify
PATH = r'D:\thesis\videos\frames'
EXTENSION = '.csv'
FILE = 'frame125_bgr_palette.csv'
#FILES = ['frame250.jpg', 'frame375.jpg']     # for a list of images to process 
# load files from directory 
FILES = ['frame12625_bgr_palette.csv', 'frame125_bgr_palette.csv']
for r, d, f in os.walk(PATH): # r=root, d=directories, f = files
    for file in f:
        if EXTENSION in file:
            FILES.append(file) 
            
# filters
FILTER_CP = None  # None
DEPTH = 'row 19' # ['row 0','row 19'] (top-to-bottom hierarchy)
FILTER_COLOR = 'azurblau' # basic color (desired format: lab)
THRESHOLD_RATIO = [] #%color pixel

FILTER_HUE = 'blue'  # [0-360°], hue_range['red']['pure'], hues = [red, orange, yellow, green, cyan, blue, violet, magenta]
FILTER_SAT = 0.4 # [0,1]  0 - gray, 100 - color (saturation)
FILTER_VAL = 1 # [0,1]  0 - black, 100 - color (value/brightness), 0 - black, 50 - color, 100 - white (lumination)



#%%

### Processing ###

# convert color 
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

# convert numpy array of colors
def convert_array(nparray, origin, target='RGB'): 
    """helper function: convert_color """
    # convert to RGB
    converted_colors = []
    for col in nparray: 
        if origin == 'BGR' and target == 'RGB':        
            converted_color = convert_color(col, cv2.COLOR_BGR2RGB)
        if origin == 'LAB' and target == 'RGB':     
            converted_color = convert_color(col, cv2.COLOR_LAB2RGB)*255
        if origin == 'HSV' and target == 'RGB':     
            converted_color = convert_color(col, cv2.COLOR_HSV2RGB)*255
        if origin == 'RGB' and target == 'HSV':     
            converted_color = convert_color(col, cv2.COLOR_RGB2HSV)
        converted_colors.append(converted_color)
    return converted_colors

# display color palette as bar 
def display_color_grid(palette, origin='RGB', colorbar_count=10):
    """helper function: convert_array, convert_color """ 
    if origin == 'BGR':
        rgbcolors = convert_array(palette, 'BGR')
    if origin == 'LAB': 
        rgbcolors = convert_array(palette, 'LAB')
    if origin == 'HSV': 
        rgbcolors = convert_array(palette, 'HSV')
    x= 0
    for r in rgbcolors: 
        if len(rgbcolors[x:x+colorbar_count]) == colorbar_count:
            palette = np.array(rgbcolors[x:x+colorbar_count])[np.newaxis, :, :]
            plt.figure(figsize=(colorbar_count*2,5))
            plt.imshow(palette.astype('uint8'))
            #plt.imshow(palette)
            plt.axis('off')
            plt.show()
            x += colorbar_count
        else: 
            if x == len(palette): 
                break
            else: 
                palette = np.array(rgbcolors[x:])[np.newaxis, :, :]
                plt.figure(figsize=(colorbar_count*2,2))
                plt.imshow(palette.astype('uint8'))
                plt.axis('off')
                plt.show()
                break
        return rgbcolors
            


#%%
# color range detection for 8 color categories

# manually-adjusted color classification chart 
hue_range = {'red': {'pure': 355, 'range': [341,10]}
            , 'orange': {'pure': 25, 'range': [11,40]}
            , 'yellow': {'pure': 50, 'range': [41,60]}
            , 'green': {'pure': 115, 'range': [61,170]}
            , 'cyan': {'pure': 185, 'range': [171,200]}
            , 'blue': {'pure': 230, 'range': [201,250]}
            , 'violet': {'pure': 275, 'range': [251,290]}
            , 'magenta': {'pure': 315, 'range': [291,340]}
            }

satval_range = {'gray': {'pure': [0,50]} # range defined by func get_gray
            , 'white': {'pure': [0,0]} # range defined by func get_white
            }
val_range = {'black': {'pure': 0, 'range': [0,20]}}

def get_white(number, enter='sat2'): 
    if enter == 'sat2': 
        assert number >= 0 and number <= 5, "Saturation needs to be between 0 and 5."
        value = number + 95
        return value 
    elif enter == 'value': 
        assert number >= 95 and number <= 100, "Value needs to be between 95 and 100."
        sat2 = number - 95 
        sat1 = 0
        return [sat1, sat2]
    else: 
        print("The number cannot be a hsv-white determinant.")


def get_gray(number, enter='sat2'): 
    if enter == 'sat2':
        assert number <= 100 and number >= 0, "Saturation needs to be between 0 and 100."
        value = -.75*number + 95
        assert value <=95 and value >= 20 
        return value 
    elif enter == 'value': 
        assert number <=95 and number >= 20, "Value needs to be between 20 and 95."
        sat2 = (number - 95) / -.75
        if sat2 ==-0.0:
            sat2 = 0 
        assert sat2 <= 100 and sat2 >= 0
        sat1 = 0
        return [sat1, sat2]
    else: 
        print("The number cannot be a hsv-gray determinant.")

number = get_gray(22, enter='value') #0- 95; 100-20
print(number)

number = get_white(99, enter='value')
print(number)


#%%
# load function
                
def load_palette(path, file):     
    # set directory 
    os.chdir(path)
    # load data
    palette = pd.read_csv(file, index_col=0)
    return palette
    
 
def show_palette(palette, depth): 
    # define palette
    CP_subset = palette.loc['bgr_colors'][depth]
    # convert type 
    CP_subset = eval(CP_subset) # str2list
    CP_subset = np.array(CP_subset) # list2numpy array
    # show palette
    rbgcolors = display_color_grid(CP_subset, 'BGR')
    # analyze palette
    print('Number of colors in palette: ', len(CP_subset))
    return CP_subset, rgbcolors

#%%
# all palettes    
cp_pool = []

# load palette
for FILE in FILES: 
    palette = load_palette(PATH, FILE) 
    cp_pool.append(palette)
    
# show palette
#cp_row, rgb = show_palette(cp_pool[0], DEPTH)

#%%
# get hsv values for palette
def get_hsv4bgr(palette): 
    bgr_array = np.array(eval(palette.loc['bgr_colors'][-1]))
    rgb_array = convert_array(bgr_array, origin='BGR', target='RGB')
    hsv_array =[]
    for i in rgb_array:
        rgb = np.array(i)
        rgbi = np.array([[rgb/ 255]], dtype=np.float32)
        hsv = cv2.cvtColor(rgbi, cv2.COLOR_RGB2HSV)
        hsv = hsv[0, 0]
        hsv_array.append(hsv)
    hsv_list = [list(i) for i in hsv_array]
    return hsv_list 

def add_hsv2palett(hsv_list):
    # add hsv values to palette
    palettini = pd.DataFrame()
    palettini['hsv_colors'] = hsv_list
    palettini['hue'] = [i[0] for i in hsv_list]
    palettini['sat'] = [i[1] for i in hsv_list]
    palettini['val'] = [i[2] for i in hsv_list]
    palettini['hsv'] = palettini[['hue', 'sat', 'val']].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
    return palettini

palettinis = []
for palette in cp_pool: 
    hsv_list = get_hsv4bgr(palette)
    palettini = add_hsv2palett(hsv_list)
    palettinis.append(palettini)
    

#%%

# filter colors 

def filter_palette(index, palettini, palett_name, FILTER_CP=None, FILTER_HUE=None):
    # filter by color palette name 
    if FILTER_CP: 
        paletti = palettini.loc[FILTER_CP] 
    else: 
        paletti = palettini
    
    # filter by hue  
    #palett = paletti[paletti['hue'] >= hue_range[FILTER_HUE]['pure']]
    if FILTER_HUE == 'red': 
        palett1 = paletti[paletti['hue'] >= hue_range[FILTER_HUE]['range'][0]]
        palett2 = paletti[paletti['hue'] <= hue_range[FILTER_HUE]['range'][1]]
        palett = pd.concat([palett1, palett2])
    else: 
        palett = paletti[paletti['hue'].between(hue_range[FILTER_HUE]['range'][0],hue_range[FILTER_HUE]['range'][1])] # check any color except red, black, gray and white
        try: 
            palett = paletti[paletti['val'].between(val_range[FILTER_HUE]['range'][0],val_range[FILTER_HUE]['range'][1])] # check black
        except: 
            pass
        if FILTER_HUE == 'gray': 
            palett = paletti[paletti['val'][0] <= get_gray(paletti['sat'][0]*100, enter='sat2')]
                     
        
        #palett = palett[palette['sat'] <= FILTER_SAT]
        #palett = palett[palette['val'] == FILTER_VAL]
     
    # restructure type 
    palett = [eval(l) for l in palett['hsv']]
    if palett == []: 
        print(f"No {FILTER_HUE} colors found in {index+1}. palette '{palett_name}'.") 
    # convert type    
    palet = np.array(palett) # list2numpy array
    
    # sort colors 
    try: 
        #palet = palet[palet[:,0].argsort()] # sort numpy array by first element
        palet = palet[palet[:,2].argsort()]
        palet = palet[palet] # pd2list  
    except:
        pass
    
    return palet 

palet_names = [f[:-4] for f in FILES]  
print(f"Searching a total of {len(palet_names)} palettes. ")
print("Searching your chosen color in palettes: \n", ', '.join(palet_names), '.')

palets = []
for i, palettini in enumerate(palettinis): 
    palet = filter_palette(i, palettini, palet_names[i][:-12], FILTER_CP=FILTER_CP, FILTER_HUE=FILTER_HUE)
    palets.append(palet)
    
#%%
# waterfall 
# show found palette
    
if palets[0].tolist() == []: 
    print(f"No palettes found for {FILTER_HUE}.")
else: 
    for i, palet in enumerate(palets):
        print("")
        print(f"{i+1}. {palet_names[i]}")
        #display_color_grid(palet, 'HSV')
        
# analyze palette
#print(f'Number of {FILTER_HUE} colors in filtered palette: ', len(palett))    
#print(f'Number of colors in palette: ', len(paletti))    

#%%

# TODO: get color tags for a palette (quickens search)