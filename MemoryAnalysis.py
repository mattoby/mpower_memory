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

agecutoff = 45

#######################################################

#### RUN A LOOP:

# simplified code when not sample balancing:
# (with other memory features)
# take only one record per patient to remove duplicates. mean is better, but that's later..
features = fcats['game'] + ['hasParkinsons'] + fcats['demographic'] + fcats['phone']
features.remove('smartphone')
features.remove('gender')
features.remove('age')
MLexcludecols = []
labelcol = 'hasParkinsons'

# aggregate lists:
models = []
train_accs = []
y_tests = []
y_pred_probas = []

nIters = 10
for iter in range(nIters):
    print 'starting iter ', iter
    # redo data so i take 1 sample of each patient.
    grouped = data.groupby('healthCode')
    datasample = grouped.apply(lambda x: x.sample(n=1))
    # remove young patients:
    datasample = datasample[datasample['age']>agecutoff]
    # run model:
    outs = mt.build_ML_model(datasample, features, labelcol=labelcol, toPlot=[0,0,0], toPrint=False, MLexcludecols=MLexcludecols)
    model, fdf, X, y, X_names, y_name, X_train, X_test, y_train, y_test, train_acc, test_acc, rand_acc, y_pred, y_pred_proba = outs
    # capture results:
    models.append(model)
    train_accs.append(train_acc)
    y_tests.append(y_test)
    y_pred_probas.append(y_pred_proba)



######## figure out how to plot the thing:

##from sklearn.cross_validation import KFold

# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
#cv = KFold(len(y), n_folds=3)
#

plot_auc_curves_with_mean()
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

#plt.figure(figsize=(16,12))
plt.figure()

for iter in range(nIters):
#for i, (train, test) in enumerate(cv):
   #probas_ = model2.fit(X[train[0]:train[-1]], y[train[0]:train[-1]]).predict_proba(X[test[0]:test[-1]])
   probas = y_pred_probas[iter]
   y_test = y_tests[iter]

   # Compute ROC curve and area the curve
   fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, probas)
#   fpr, tpr, thresholds = roc_curve(y[test[0]:test[-1]], probas_[:, 1])
   mean_tpr += interp(mean_fpr, fpr, tpr)
   mean_tpr[0] = 0.0
   roc_auc = sklearn.metrics.auc(fpr, tpr)
   plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (iter, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Chance')

#mean_tpr /= len(cv)
mean_tpr /= nIters
mean_tpr[-1] = 1.0
mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
        label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=22)
plt.ylabel('True Positive Rate', fontsize=22)
plt.title('Receiver Operating Characteristic with Cross Validation', fontsize=22)
plt.legend(loc="lower right")
plt.show()



