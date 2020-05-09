# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:11:42 2020

@author: lsamsi
"""
# load modules
import os
import pandas as pd


### Color-Survey ###

PATH = r'D:\thesis\color-survey'
FILE = 'satfaces.xlsx'

# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ")
data.tail()
data['thesaurus'].value_counts()

#%%

### Color-Thesaurus EPFL ###

PATH = r'D:\thesis\color-thesaurus-epfl'
FILE = 'ColorThesaurus.xlsx'
OUTPUT_FILE = 'VIANHuesColorThesaurus.xlsx'

# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ")
data.info()
data = data.dropna()


datalst = data['english name'].str.split().tolist()
datalstt = [l[-1] for l in datalst]

data['english name'] = datalstt

data['english name'].value_counts()

#%%

# 28 Vian colors 
vian_hues = [
        'blue'
        , 'cyan'
        , 'green'
        , 'magenta'
        , 'orange'
        , 'pink'
        , 'red'
        , 'yellow'
        , 'beige'
        , 'black'
        , 'brown'
        , 'copper'
        , 'cream'
        , 'gold'
        , 'grey'
        , 'purple'
        , 'rust'
        , 'silver'
        , 'white'
        , 'amber'
        , 'lavender'
        , 'sepia'
        , 'apricot'
        , 'bronze'
        , 'coral'
        , 'peach'
        , 'ultramarine'
        , 'mustard'
        ]

# before preprocessing: 

flt1 = data[data['english name'].isin(vian_hues)]
flt1['english name'].value_counts()
flt1.shape[0]


#%%

# preprocessing 

# recodings         
# recode blue: blueberry, bluish, darkblue, lightblue -> blue 
data['english name'] = data['english name'].replace(['blueberry', 'bluish', 'darkblue', 'lightblue'], ['blue', 'blue', 'blue', 'blue'])
# recode blue: blueberry, bluish, darkblue, lightblue -> blue 
data['english name'] = data['english name'].replace(['brownish'], ['brown'])
# recode green: darkgreen, greenish -> green
data['english name'] = data['english name'].replace(['darkgreen', 'greenish'], ['green', 'green'])
# recode grey: greyish -> grey 
data['english name'] = data['english name'].replace(['greyish'], ['grey'])
# recode lavender: lavendar -> lavender
data['english name'] = data['english name'].replace(['lavendar'], ['lavender'])
# recode orange: orangeish -> orange 
data['english name'] = data['english name'].replace(['orangeish'], ['orange'])
# recode pink: pinkish, pinky -> pink
data['english name'] = data['english name'].replace(['pinkish', 'pinky'], ['pink', 'pink'])
# recode red: reddish -> red
data['english name'] = data['english name'].replace(['reddish'], ['red'])
# recode yellow: yellowish -> yellow 
data['english name'] = data['english name'].replace(['yellowish'], ['yellow'])


# border colors: bluegreen, bluegrey, greenblue, greyblue, orangered, yellowgreen




#%%

# after preprocessing: 

flt2 = data[data['english name'].isin(vian_hues)]
flt2['english name'].value_counts()
better = round(((flt2.shape[0] -flt1.shape[0]) / flt1.shape[0])*100,2)

print('Preprocessing will yield a {} % increase in the number of rows.'.format(better))

#%%

# save data 
flt2.to_excel(OUTPUT_FILE)

