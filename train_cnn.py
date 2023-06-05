import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import warnings
from glob import glob
from tqdm import tqdm
warnings.filterwarnings("ignore")

DATA_ROOT = 'dataset_small'
data_paths = dict()
dataframes = dict()
image_paths = dict()
magnifications = ['global', 'local']
for magnification in magnifications:
    data_paths[magnification] = [DATA_ROOT+os.sep+magnification+os.sep+'images',
                        DATA_ROOT+os.sep+magnification+os.sep+
                        magnification+'_labels.csv']
    df = pd.read_csv(data_paths[magnification][1])
    unnamed_col = df.columns[df.columns.str.contains('unnamed',case = False)]
    df.drop(unnamed_col, axis = 1, inplace = True)
    dataframes[magnification] = df
    image_paths[magnification] = glob(data_paths.get(magnification)[0] + os.sep
                                      + "*.jpg")
    print("Number of "+magnification+" images:")
    print(len(image_paths.get(magnification)))