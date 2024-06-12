from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def min_max_scaler(images: list) -> list:
    return MinMaxScaler().fit_transform(images)

def black_and_white_scaler(images: list) -> list:
    
    def transform_value(x):
        if (x < 128):
            return 0
        else:
            return 1  
        
    def mapper(images):
        return list(map(transform_value, images))
          
    return list(map(mapper, images))


