
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

# adjustment for different systems (share.io PosixPath)
# plt = platform.system()
# if plt == 'Linux': 
#     pathlib.WindowsPath = pathlib.PosixPath

## Layout App
##################

st.title('**Balinese Dance Style** - Classification')
st.markdown("""
### AI - Computer Vision Recognition with **fastai/pytorch**

Classifing images between the most famous balinese dances  \n \n 
*Kecak* , *Barong*  &  *Legong*

2021.03.21: Accuracy on ResNet34 Architectur: 80% after 5 Epochs
\n
""")
link1 = 'Model & Data Preprocessing [Github](https://github.com/Svensone/fastAI2-Bali-Dance/blob/main/2021_03_29_%5Bfastaiv2%5D_New_Balinese_Dance_Image_Recognition.ipynb)'
link2 = 'Deployment [Github](https://github.com/Svensone/fastAI2-Bali-Dance)'
st.markdown(link1, unsafe_allow_html=True, )
st.markdown(link2, unsafe_allow_html=True)

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

PATH = pathlib.Path(__file__).parent
IMG_PATH = PATH.joinpath("assets").resolve()

set_png_as_page_bg(IMG_PATH.joinpath('bg2.jpeg'))

#######################################
### Image Classification
#######################################

def prediction(img, display_img):
    # display the image
    st.image(display_img, use_column_width=True)

    # loading spinner
    with st.spinner('Wait a second .....'):
        time.sleep(3)

    # get data
    DATA_PATH = PATH.joinpath('data')
    CSV_PATH = DATA_PATH.joinpath('cleaned.csv')
    MODEL_PATH  = DATA_PATH.joinpath('models', 'v2-stage-1.pth')

    data = ImageDataLoaders.from_csv(path=DATA_PATH , csv_fname='cleaned.csv', valid_pct=0.2, item_tfms=Resize(224), csv_labels='cleaned.csv', bs=64)

#  load Learner
    learn = cnn_learner(data, models.resnet34, metrics=accuracy)
    learn.load('v2-stage-1')

    # Prediction on Image
    predict_class = learn.predict(img)[0]
    predict_proba = learn.predict(img)[2]
    
    print(predict_proba)
    print(f'predict class {predict_class}')

    proba = float(predict_proba[1]) if str(predict_class) == 0 else float(predict_proba[0])
    proba = (proba * 100)
    proba = int(proba)

    # Display results
    if str(predict_class) == 'legong':
        st.success(f'This is a Scene of the famous Legong Kraton Dance. Probability of Prediction is {proba} % ')
        link = 'Find out more [Wikipedia](https://en.wikipedia.org/wiki/Legong)'
        st.markdown(link, unsafe_allow_html=True)
    elif str(predict_class) == "barong":
        st.success(f'Probability of Prediction is {proba} %, This is a Scene of the Barong Dance, which is together with Sanghyang considered to be an ancient native Balinese Dance')
        link = '[Barong Wikipedia](https://en.wikipedia.org/wiki/Barong_(mythology)#Barong_dance)'
        st.markdown(link, unsafe_allow_html=True)
    else:
        st.success(f'This is a Scene of the Kecak Dance, created by german artist Walter Spies in 1930s. Probability of Prediction is {proba} % ')
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

    TEST_IMG_PATH = PATH.joinpath('test', test_img)
    file_path = 'test/'+ test_img

    img = PILImage.create(TEST_IMG_PATH)
    # print(img)

    ##### TEST
    ################
    im_test3 = PIL.Image.open(TEST_IMG_PATH)
    display_img = np.asarray(im_test3) # Image to display

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

            # call predict func
            prediction(tpil, display_img)
        except:
            st.text("Invalid URL")
