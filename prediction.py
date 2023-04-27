import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value


def scaler(v):
    return StandardScaler().fit_transform(np.array([v]).reshape(1, -1)) 


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)