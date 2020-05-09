# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:26:44 2020

@author: Linda Samsinger

Download Images form Google Image Search using an API 

"""



# to specify 
SEARCHKEY = 'ultramarine color'
MAXLINKS = 100 
MARGIN = 10 # margin for corrupt images
TOFETCH = MAXLINKS + MARGIN 
FOLDER_PATH = r'D:\thesis\images\google'


#%%

# get image urls from browser 

import time 
import io
from selenium import webdriver

DRIVER = webdriver.Chrome('C:/Users/Anonym/Desktop/thesis/Lindner_CIC20_ColorThesaurus/chromedriver.exe')

def fetch_image_urls(query, max_links_to_fetch, wd, results_start=0, sleep_between_interactions=1):
    
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    
    
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = results_start
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            return
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)
        
    wd.close()
    image_urls = list(image_urls)
#    if len(image_urls) != max_links_to_fetch:       
#        image_urls = image_urls[:max_links_to_fetch]
    return image_urls


urls = fetch_image_urls(SEARCHKEY, MAXLINKS2FETCH, DRIVER,1)

#%% 

# persist image

import os
import requests
from PIL import Image
import hashlib

def persist_image(folder_path:str,url:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url}")
        return url 
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")
        return 0

# saves images to folder (unsorted)
savedimages = []
for i in range(len(urls)): 
    print(i+1)
    url = persist_image(FOLDER_PATH, urls[i])
    if url != 0: 
        savedimages.append(url)
    if len(savedimages) == MAXLINKS: 
        break

print("---")
print(f"SUCCESS - {len(savedimages)} images saved into {FOLDER_PATH}")
print("---")