import itertools
import numpy as np
import pandas as pd
import random as r
import os
from scipy.spatial.distance import pdist
import shutil

base = "./"
to = "./filtered_train/"

df = pd.read_csv("./filtered_train.csv")
for _, row in df.iterrows():
    fn = row['image_name']+'.png'
    shutil.copy2(base + fn , to + fn)


to = "./filtered_valid/"

df = pd.read_csv("./filtered_valid.csv")
for _, row in df.iterrows():
    fn = row['image_name']+'.png'
    shutil.copy2(base + fn , to + fn)