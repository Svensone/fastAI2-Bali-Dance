
# from fastai.learner import load_learner
from fastai.basics import *
from fastai.vision.all import load_image, PILImage, Image, ImageDataLoaders, cnn_learner, models, accuracy
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
import PIL.Image

from helper import local_css

# uncomment for development on local Windows 
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title('**Balinese Dance Style** - Classification')
st.markdown("""
### AI - Computer Vision Recognition with **fastai/pytorch**

Classifing images between the most famous balinese dances  \n \n 
*Kecak*,*Barong*& *Legong*
""")

## test local css
##################
local_css("style.css")


# Set Background Image *local file" - heroku ? working
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
    
    # load_learner() not working, need .load() 
    #  setup model
    path_data = Path('/data')
    data = ImageDataLoaders.from_csv(path_data, csv_fname='cleaned.csv', valid_pct=0.2, item_tfms=Resize(224), csv_labels='cleaned.csv', bs=64)
    learn = cnn_learner(data, models.resnet34, metrics=accuracy)
    learn.load('v2-stage-1')

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

    ## join with os.path.joinpath() or PurePosixPath
    ############## Python uses on OS/Linuz PosixPath, on Windows 'WindowsPath'
    # uncomment above temp.pathlib
    print(test_img)
    file_path = 'test/'+ test_img
    print(file_path)
    # file_path = 'test/' + test_img
    # Parse Image for clf
    img = PILImage.create(file_path)

    # get the image display
    # display_img = mpimg.imread(file_path)

    ##### TEST
    ################
    im_test3 = PIL.Image.open(file_path)
    display_img = np.asarray(im_test3) # Image to display

    # call predict func with this img as parameters
    prediction(img, display_img)

## Predition from URL Image not yet working - converting to fastAI Image object error
##################################################
else:
    url = st.text_input('URL of the image')
    if url !='':
        try:
            # Read image from the url
            im = PIL.Image.open(requests.get(url, stream=True).raw)

            display_img = np.asarray(im) # Image to display
            ## Transform PIL Image to pytorch tensor
            min_img_size = 224
            transform_pipeline = T.Compose([T.Resize(min_img_size), T.ToTensor()])
            img_tensor = transform_pipeline(im)
            # adjust dims for fastai clf
            img_unsqueeze = img_tensor.unsqueeze(-1)
            #convert to fastAi Image object
            img_fastai = Image(img)
            # call predict func
            prediction(img_fastai, display_img)
        except:
            st.text("Invalid URL")
