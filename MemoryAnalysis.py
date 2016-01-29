# this file is where i'm playing with new methods..

import synapseclient
from synapseclient import Project, Folder, File
import pandas as pd
import json
import pickle
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import memorytools as mt
# from pandas import DataFrame, Series
# if running in ipython... (so i can see plots)
%pylab
# for reloading modules:
%load_ext autoreload
# %autoreload 2 # this will reload all modules


#############
## startup ##
#############

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

# look at individual game results:
#memrecordId = '5a0b4204-8a6c-430f-be93-c5aa2d6c9e33'
#avg_features_by_sizes, games, games_by_sizes = mt.form_features_from_memory_record(filePaths, data, memrecordId)


##############
############## exploration:
##############


####################
## model building ##
####################
# sklearn:
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


### run_entire_logistic_regression(data, features) # build this



##################### Preprocess data for machine learning:
# define features:
#features_df = data[["game_score","age","game_numFails", "phoneInfo",
#    "education", "gender", "phoneUsage", "smartphone", "hasParkinsons"]]
features_df = data[["game_score","age","game_numFails", "phoneInfo",
    "education", "gender", "phoneUsage", "smartphone", "hasParkinsons"]]

features_df = data[['game_score', 'hasParkinsons']]

features_df = mt.convert_features_to_numbers(features_df)
features_df = mt.move_col_to_end_of_df(features_df, 'hasParkinsons')

# do more processing here, in case of features with lots of nas

# drop na rows:
features_df = features_df.dropna()

# convert to matrices for machine learning:
labelcol = 'hasParkinsons'
X, y, X_names, y_name = mt.convert_features_df_to_X_and_y_for_machinelearning(features_df, labelcol)

##################### Perform machine learning:

# do cross validation manually:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# scale features:
stdsc = StandardScaler()
stdsc.fit(X_train)
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# create logistic regression model:
lr = LogisticRegression(C=1000.0, random_state=0)##
#lr = linear_model.LogisticRegression(penalty='l1', C=0.1) # with regularization
lr.fit(X_train_std, y_train)

# assess regression performance:
lr.coef_
lr.intercept_ # this is the 0 coeff?
lr.fit(X_train_std, y_train)
print 'training accuracy:', lr.score(X_train_std, y_train)
print 'test accuracy:', lr.score(X_test_std, y_test) # suspiciously high..
lr.intercept_
lr.coef_ # only using 4 features.. which ones?
# mt.plot_decision_regions(X_combined_std, y_combined_std, classifier=lr, test_idx=range(len(X_train_std),len(X_combined_std)+1))
X_names_heavy = X_names[np.where(np.abs(lr.coef_) > 0.1)[1]]
Scoef = convert_regression_coefs_to_pdSeries(lr.coef_, X_names)
print Scoef.sort_values


## second round:

















##############
## old/junk ##
##############
for col in data.columns:
    if col in features:
        print col



sortinds = lr.coef_.argsort()
df = pd.DataFrame(lr.coef_.reshape(8, -1))
coef = lr.coef_.reshape(8, -1)
outdf = pd.Series([lr.coef_)



outdf = pd.DataFrame(data={'coef':[lr.coef_]}, index=X_names)
outdf = pd.Series([lr.coef_], index=X_names)

print [lr.coef_[sortinds], lr.X_names[sortinds]]

sorted_arr1 = arr1[arr1inds[::-1]]
sorted_arr2 = arr2[arr1inds[::-1]]

lrcoefinds = lr.coef_.argsort()
X_names[lrcoefinds[::-1]]
lr.coef_.T[lrcoefinds[::-1]]
subplot
figure()

plot(data['age'],data['hasParkinsons'])
d1 = [data['age'],data['hasParkinsons']]
boxplot(d1)



#X_train_norm = mms.fit_transform(X_train)
#X_test_norm = mms.transform(X_test)

# fit StandardScaler only once on the training data, then use those params to transform the test set or any new datapoint











## chop features that are weighted too highly...
# from unchopped, weighted features are: array([u'smartphone', 'employment_2', 'maritalStatus_1', 'smartphone_3'], dtype=object)

#suspicious_features = ['employment_2', 'maritalStatus_1', 'employment_0',
#    'employment_1', 'employment_3', 'employment_4',
#    'employment_5', 'employment_6', 'employment_7', 'maritalStatus_0',
#    'maritalStatus_2', 'maritalStatus_3', 'maritalStatus_4',
#    'maritalStatus_5', 'maritalStatus_6']
#features_df = features_df.drop(suspicious_features, axis=1)



#y = features_df['hasParkinsons'].astype('int').tolist()
#features_df = features_df.drop('hasParkinsons',1)
#X = features_df.as_matrix()


##############

# First, do only with age & score!!!??


# do cross validation manually:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# retrain on the full dataset at end.

# scaling
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#mms = MinMaxScaler()
stdsc = StandardScaler()
#X_train_norm = mms.fit_transform(X_train)
#X_test_norm = mms.transform(X_test)
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
# fit StandardScaler only once on the training data, then use those params to transform the test set or any new datapoint

# L1 - sparse feature vectors (most weights 0)
# L2 - penalize large individual weights
# use L1 (i.e., form of feature selection)

lr = linear_model.LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print 'training accuracy:', lr.score(X_train_std, y_train)
print 'test accuracy:', lr.score(X_test_std, y_test) # suspiciously high..
lr.intercept_
lr.coef_ # only using 4 features.. which ones?


X_names_heavy = X_names[np.where(lr.coef_ > 0)[1]]
X_names_heavy

arr1inds = arr1.argsort()
sorted_arr1 = arr1[arr1inds[::-1]]
sorted_arr2 = arr2[arr1inds[::-1]]

lrcoefinds = lr.coef_.argsort()
X_names[lrcoefinds[::-1]]
lr.coef_.T[lrcoefinds[::-1]]
subplot
figure()

plot(data['age'],data['hasParkinsons'])
d1 = [data['age'],data['hasParkinsons']]
boxplot(d1)

#!!!!! X_names[np.sort(lr.coef_)]
# sort X_names.sort(lr_coef_) # doesn't work!!!!!!!!!!!!!

# encode education as #'s not categorical?
































# do cross validation models ==>
# https://civisanalytics.com/blog/data-science/2015/12/17/workflows-in-python-getting-data-ready-to-build-models/
clf1 = linear_model.LogisticRegression()
score1 = sklearn.cross_validation.cross_val_score( clf1, X, y , cv=5)
print( score1 )
​
clf2 = sklearn.tree.DecisionTreeClassifier()
score2 = sklearn.cross_validation.cross_val_score( clf2, X, y , cv=5)
print( score2 )

clf3 = sklearn.ensemble.RandomForestClassifier()
score3 = sklearn.cross_validation.cross_val_score( clf3, X, y , cv=5)
print( score3 )

import sklearn.feature_selection
​
select = sklearn.feature_selection.SelectKBest(k=10)
selected_X = select.fit_transform(X, y)
​
print( selected_X.shape )

clf1a = linear_model.LogisticRegression()
score1a = sklearn.cross_validation.cross_val_score( clf1a, selected_X, y )
print( score1a )

#These aren't fitted??


logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, y)

h = .02  # step size in the mesh
# normalize before running logistic regression?

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])





#########http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])






















# transform categorical features
def transform_feature(df, column_name ):
    '''
    Transforms categorical features from dataframe df & column_name into numbers
    From https://civisanalytics.com/blog/data-science/2015/12/17/workflows-in-python-getting-data-ready-to-build-models/
    '''
    unique_values = set( df[column_name].tolist() )
    transformer_dict = {}
    for ii, value in enumerate(unique_values):
        transformer_dict[value] = ii

    def label_map(y):
        return transformer_dict[y]
    df[column_name] = df[column_name].apply( label_map )
    return df


def hot_encoder(df, column_name):
    '''
    Hot encodes categorical feature (that is already converted to #'s by transform_feature) into multiple binary columns

    From: https://civisanalytics.com/blog/data-science/2015/12/23/workflows-in-python-curating-features-and-thinking-scientifically-about-algorithms/

    Look at instead: pd.get_dummies(df[['col1','col2','col3']]) # will do the encoding instead..
    '''
    column = df[column_name].tolist()
    column = np.reshape( column, (len(column), 1) )  ### needs to be an N x 1 numpy array
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit( column )
    new_column = enc.transform( column ).toarray()
    column_titles = []
    ### making titles for the new columns, and appending them to dataframe
    for ii in range( len(new_column[0]) ):
        this_column_name = column_name+"_"+str(ii)
        df[this_column_name] = new_column[:,ii]
    return df


def hot_encode_categorical_features(features_df, columns_to_transform):
    '''
    Transforms and hot-encodes categorical feature into multiple binary columns
    Columns_to_transform is a list of column names. e.g., :
    ["phoneInfo", "smartphone"]
    '''

    for column in columns_to_transform:
        features_df = transform_feature( features_df, column )
        features_df = hot_encoder( features_df, column)
#        features_df = pd.get_dummies
    print( features_df.head() )
    return features_df


