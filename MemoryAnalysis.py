## Startup:

import synapseclient
from synapseclient import Project, Folder, File
import pandas as pd
import json
import pickle
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
import seaborn as sns
import os
from numpy import nan
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import ranksums
%pylab
%load_ext autoreload
%autoreload 2
#%matplotlib inline


## set options ##

sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

# so i can see all rows of dfs
pd.set_option('display.max_columns', 500)

# so that i can print as many lines as i want
np.set_printoptions(threshold='nan')

## import my memorytools module ##

import memorytools as mt


# Load up the memory & demographic data:

# initialize environment:
synuser = os.environ['SYNAPSE_USER']
synpass = os.environ['SYNAPSE_PASS']
mt.loadSynapseRecordsFromScratch = False
syn, memory, memorysyn, filePaths, demographics, demosyn, data = mt.create_memory_environment(synuser, synpass)
data = mt.filter_data_for_popular_phones(data)

# pull out features from games:
fromFile = True#False
toSave = False#True
data = mt.add_memory_game_features_to_data(filePaths, data, fromFile = fromFile, toSave=toSave, outFileName='memory_data_with_features.p')

# add memory composite features:
data = mt.add_composite_features_to_data(data)


from sklearn import linear_model
import sklearn
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.tree
import sklearn.ensemble
import numpy as np
from sklearn.utils.validation import check_consistent_length, _num_samples
import sklearn.preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# load feature categories
fcats = mt.define_feature_categories()

from memorytools import *


grouped.groups
grouped.mean().head(2)


data2 = data.copy()
grouped = data2.groupby('healthCode')

for healthCode, group in grouped:
    group = group.dropna(subset=features)
    newthing = group.mean()

for healthCode, group in grouped:
    print group
    print healthCode


# take 1, not mean. improve to mean later.
print '\n\n\n'
print df
print '\n'
features = ['c','b']
for healthCode, group in grouped:

    group = group.dropna(subset=features)
#    groupmean = group.mean(numeric_only=False)
    group_sample = group.sample(n=1)

    print group
    print '\n'
    print group_sample
#    print groupmean
    print '\n'

grouped = data.groupby('healthCode')
data2 = grouped.apply(lambda x: x.sample(n=1))

mean_with_strings()



# within a group
loop through feature columns
if numerical, take mean
if non-numerical, take 1st value.













