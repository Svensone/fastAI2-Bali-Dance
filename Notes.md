
Set up:
- conda activate fastai2 env


fastai ML models: since bigger than 100mb (github limit)
- track with 
git lfs track "model.pkl"
git add .gitattributes

2021.03.13:
- problem with PosixPath in Production Modus on share.io (https://share.streamlit.io/svensone/fastai2-bali-dance/main/app.py)
- on local machine works fine with :

data = ImageDataLoaders.from_csv(path='data/', csv_fname='cleaned.csv', valid_pct=0.2, item_tfms=Resize(224), csv_labels='cleaned.csv', bs=64)
    learn.load('v2-stage-1')