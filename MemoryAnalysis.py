# Setup:
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
from sklearn.metrics import confusion_matrix
%load_ext autoreload
#%matplotlib inline
%pylab

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


# ML setup:

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


##############################################################
##############################################################
################### Do machine learning ######################
##############################################################
##############################################################

!!!!! # NEED TO FILTER NAS BEFORE RESAMPLING.

#### deal with age confound

# define features
fcats = mt.define_feature_categories()
features = fcats['game'] + fcats['demographic'] + ['hasParkinsons'] #+ fcats['output']
#features.remove('medTimepoint')
outs = mt.build_ML_model_age_corrected_and_samplebalanced(data, features, toPlot=True)



def build_ML_model_age_corrected_and_samplebalanced(data, features, labelcol='hasParkinsons', toPlot=False):
    '''
    Does age correction & sample balancing, then runs ML

    This function is ugly, needs to be cleaned up & generalized
    features must include the labelcol

    '''

    # define the columns to sample balance & resample on:
    distcol = 'age'
    splitcol = 'hasParkinsons'
    nbins = 10
    nResamples = 600

    # build features dataframe:
    fdf = data[features]
    fdf = mt.convert_features_to_numbers(fdf)

    # resample non-Park to same age distribution as Parkinsons:
    splitVal_resample = False
    splitVal_guide = True

    df_resampled, df_guide, df_resample = mt.resample_to_match_distribution(fdf, distcol, splitcol, splitVal_resample, splitVal_guide, nbins, nResamples)
    df_resampled_np = df_resampled
    df_Parkinsons = df_guide
    df_np = df_resample

    # resample Park to the resampled non-Park for sample balancing:
    fdf2 = df_resampled_np.append(df_Parkinsons)

    splitVal_resample = True
    splitVal_guide = False
    df_resampled, df_guide, df_resample = mt.resample_to_match_distribution(fdf2, distcol, splitcol, splitVal_resample, splitVal_guide, nbins, nResamples)
    df_resampled_Park = df_resampled

    ### Redo machine learning with these sets:
    df = df_resampled_np.append(df_resampled_Park)
    # features = fcats['game'] + ['hasParkinsons']# + fcats['demographic'] + fcats['output']

    #labelcol = 'hasParkinsons'
    #mt.display_num_nulls_per_column(df[features])

    # labelcol goes in here, and is what is learned with the model:
    features_df, X, y, X_names, y_name, X_train, X_test, y_train, y_test, stdsc, X_train_std, X_test_std, X_combined_std, y_combined = mt.prep_memory_features_for_machine_learning(df, features, labelcol, convert_features_to_nums=False)

    # create model:
    mod = RandomForestClassifier(n_estimators=100)
    #lr = linear_model.LogisticRegression(penalty='l1', C=0.1) # with regularization
    mod.fit(X_train, y_train)

    # which features matter?
    mat = mod.predict_proba(X_test)

    #  Confusion matrix:
    y_pred = mod.predict(X_test)
    sklearn.metrics.roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    if toPlot == True:
        # plot first set:
        plt.figure()
        sns.distplot(df_guide[distcol].dropna(), label='hasParkinsons')
        sns.distplot(df_resample[distcol].dropna(), label='no Parkinsons')
        sns.distplot(df_resampled[distcol].dropna(), label='no Parkinsons, resampled')
        plt.legend(loc=2)

        # test pval first set:
        x = df_resampled[distcol].dropna().values
        y = df_guide[distcol].dropna().values
        p2 = ranksums(x, y)
        print p2
#        print 'ranksum pval for age corrected = %s' % p2

        # plot second set:
        plt.figure()
        sns.distplot(df_Parkinsons[distcol].dropna(), label='hasParkinsons')
        sns.distplot(df_np[distcol].dropna(), label='no Parkinsons')
        sns.distplot(df_resampled_np[distcol].dropna(), label='no Parkinsons, resampled')
        sns.distplot(df_resampled_Park[distcol].dropna(), label='Parkinsons, resampled')
        plt.legend(loc=2)

        # test pval 2nd set:
        x = df_resampled_np[distcol].dropna().values
        y = df_resampled_Park[distcol].dropna().values
        p2 = ranksums(x, y)
        print p2
#        print 'ranksum pval for sample balanced = %s' % p2

        ###### assess performance:
        mod.fit(X_train, y_train)
        print 'training accuracy:', mod.score(X_train, y_train)
        print 'test accuracy:', mod.score(X_test, y_test)
        print 'num actual positives = %s' % sum(y)
        print 'num actual negatives = %s' % (len(y) - sum(y))
        print 'random accuracy would be %s' % (float(sum(y))/len(y))
        print '\n'

        # feature importances:

        print 'feature importances:'
        S = pd.Series(mod.feature_importances_, index=X_names, name="feature importances")
        print S.sort_values()


    return mod, features_df, X, y, X_names, y_name, X_train, X_test, y_train, y_test, stdsc, X_train_std, X_test_std, X_combined_std, y_combined











