import numpy as np
import pandas as pd

def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value

def std_scaler(value, label):
    df = pd.read_csv('data/raw/WildBlueberryPollinationSimulationData.csv')
    colum = df[label]
    res = ( (value - np.mean(colum)) / np.std(colum)  )
    return res

def log(value):
    return np.log(value)

def get_prediction(data, model):
    return model.predict(data)