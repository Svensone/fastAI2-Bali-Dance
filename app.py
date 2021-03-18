
# from fastai.learner import load_learner
from fastai.basics import *
from fastai.vision.all import *
import torchvision.transforms as T

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.image as mpimg
import requests
from io import BytesIO
from pathlib import Path
import pathlib
import base64
from PIL import Image
import PIL.Image

## get local css
##################
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
local_css("style.css")

## Layout App
##################

st.title('**Balinese Dance Style** - Classification')
st.markdown("""
### AI - Computer Vision Recognition with **fastai/pytorch**

Classifing images between the most famous balinese dances  \n \n 
*Kecak*,*Barong*& *Legong*
""")

# Set Background Image *local file"
###################################################
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = '''
        <style>
        body {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
        
        st.markdown(page_bg_img, unsafe_allow_html=True)
        return

set_png_as_page_bg('assets/bg2.jpeg')

#######################################
### Image Classification
#######################################

def prediction(img, display_img):
    # display the image
    st.image(display_img, use_column_width=True)

    # loading spinner
    with st.spinner('Wait a second .....'):
        time.sleep(3)
    
    # Setup ML-model
    # load_learner() not working, need .load() 

    # Problems with PosixPath/ Windowspath
    # path = Path('data\')
        
    # uncomment for development on local Windows 
    # temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath
    data_path = pathlib.PurePath('data')
    csv_path = pathlib.PurePath('data', 'cleaned.csv')
    model_path  = pathlib.PurePath('data', 'models', 'v2-stage-1.pth')

    data = ImageDataLoaders.from_csv(path=data_path , csv_fname='cleaned.csv', valid_pct=0.2, item_tfms=Resize(224), csv_labels='cleaned.csv', bs=64)
    
    learn = cnn_learner(data, models.resnet34, metrics=accuracy)
    learn.load( 'v2-stage-1')

    predict_class = learn.predict(img)[0]
    predict_prop = learn.predict(img)[2]

    # Display results
    if str(predict_class) == 'legong':
        st.success('this is a Scene of the famous Legong Kraton Dance')
        link = 'Find out more [Wikipedia](https://en.wikipedia.org/wiki/Legong)'
        st.markdown(link, unsafe_allow_html=True)
    elif str(predict_class) == "barong":
        st.success('this is a Scene of the Barong Dance, which is together with Sanghyang considered to be an ancient native Balinese Dance')
        link = '[Barong Wikipedia](https://en.wikipedia.org/wiki/Barong_(mythology)#Barong_dance)'
        st.markdown(link, unsafe_allow_html=True)
    else:
        st.success('this is a Scene of the Kecak Dance, created by german artist Walter Spies in 1930s')
        link = '[Kecak Wikipedia](https://en.wikipedia.org/wiki/Kecak)'
        st.markdown(link, unsafe_allow_html=True)

#######################################
### Image Selection
#######################################

option1= 'Choose a test image from list'
option2= 'Predict your own Image'

option = st.radio('', [option1, option2 ])

if option == option1:
    # Select an image
    list_test_img = os.listdir('test')
    test_img = st.selectbox(
        'Please select an image:', list_test_img)
    # Read the image
    test_img = test_img

    file_path = 'test/'+ test_img

    img = PILImage.create(file_path)
    # print(img)
    ##### TEST
    ################
    im_test3 = PIL.Image.open(file_path)
    display_img = np.asarray(im_test3) # Image to display
    print(img)
    # call predict func with this img as parameters
    prediction(img, display_img)

## Predition from URL Image not yet working - converting to fastAI Image object error
##################################################
else:
    url = st.text_input('URL of the image')
    if url !='':
        # print(url)
        try:
# test url pic
# https://volunteerprogramsbali.org/wp-content/uploads/2015/11/news-108.jpg
            # Read image from the url
            response = requests.get(url)
            pil_img = PIL.Image.open(BytesIO(response.content))
            display_img = np.asarray(pil_img) # Image to display
            
            # Transform the image
            timg = TensorImage(image2tensor(pil_img))
            tpil = PILImage.create(timg)
            print(tpil)

            # call predict func
            prediction(tpil, display_img)
        except:
            st.text("Invalid URL")
