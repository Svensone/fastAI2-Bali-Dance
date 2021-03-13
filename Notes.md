

conda activate fastai2 env

2021.03.13:
- problem with PosixPath in Production Modus on share.io (https://share.streamlit.io/svensone/fastai2-bali-dance/main/app.py)
- on local machine works fine with :

data = ImageDataLoaders.from_csv(path='data/', csv_fname='cleaned.csv', valid_pct=0.2, item_tfms=Resize(224), csv_labels='cleaned.csv', bs=64)
    learn.load('v2-stage-1')