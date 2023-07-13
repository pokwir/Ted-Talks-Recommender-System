import pandas as pd 
import numpy as np

#--------------------------------------------convert data types--------------------------------------------#

def convert_data_types(df):
    '''function to convert column data types for data processing'''
    df = df.astype({
        'author': 'string',
        'talk': 'string',
        'description': 'string',
        'likes': 'int',
        'views': 'int'
    })
    return df