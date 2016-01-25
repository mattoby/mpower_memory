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

# initialize environment:
synuser = os.environ['SYNAPSE_USER']
synpass = os.environ['SYNAPSE_PASS']
syn, memory, memorysyn, filePaths, demographics, demosyn, data = mt.create_memory_environment(synuser, synpass)

# Filter out all but the most popular 3 phones:
#d2 = data[]
numuserscutoff = 1000
phonegroups = data.groupby('phoneInfo').size()
goodphones = phonegroups[phonegroups > numuserscutoff].index
data = data[data.phoneInfo.isin(goodphones)]
print "(phones are now filtered for only the most popular ones)"

# sklearn:
from sklearn import linear_model
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.tree
import sklearn.ensemble
import numpy as np
from sklearn.utils.validation import check_consistent_length, _num_samples

# For scikit-learn part:

# transform categorical features
# from https://civisanalytics.com/blog/data-science/2015/12/17/workflows-in-python-getting-data-ready-to-build-models/
def transform_feature( df, column_name ):
    unique_values = set( df[column_name].tolist() )
    transformer_dict = {}
    for ii, value in enumerate(unique_values):
        transformer_dict[value] = ii

    def label_map(y):
        return transformer_dict[y]
    df[column_name] = df[column_name].apply( label_map )
    return df

# from: https://civisanalytics.com/blog/data-science/2015/12/23/workflows-in-python-curating-features-and-thinking-scientifically-about-algorithms/
import sklearn.preprocessing
# sklearn.preprocessing.LabelEncoder # look into instead?.. # no
# pd.get_dummies(df[['col1','col2','col3']]) # will do the encoding instead..
def hot_encoder(df, column_name):
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

##################### Preprocess:
# features_df.isnull().sum() # looks at the nulls in each col
#education - 8 nans
#employment - 19 nans
#gender - 5 nans
#lastSmoked - >5000 nans
#maritalStatus - 18 nans
#phoneUsage - 6 nans
#smartphone - 5 nans
# age - 15 nans

#"brainStim",
# Last 'feature' is output variable, to be chopped into y
# Continuous features are listed first
features_df = data[["game_score","age","game_numFails",
    "phoneInfo","education",
    "employment","gender","maritalStatus",
    "phoneUsage", "smartphone", "hasParkinsons"]]

#"brainStim", "education"
names_of_columns_to_transform = ["phoneInfo",
    "education","employment",
    "gender","maritalStatus",
    "phoneUsage", "smartphone"] # fix so that parkinson's column is treated correctly!!!!
# diagYear... onsetYear

for column in names_of_columns_to_transform:
    features_df = transform_feature( features_df, column )
    features_df = hot_encoder( features_df, column)
print( features_df.head() )

# put parkinsons column at end:
y_df = features_df['hasParkinsons']
features_df = features_df.drop('hasParkinsons', axis=1)
features_df['hasParkinsons'] = y_df

# drop nas:
# notnanrows = np.where(features_df['age'].notnull())[0]
features_df = features_df.dropna()

## chop features that are weighted too highly...
# from unchopped, weighted features are: array([u'smartphone', 'employment_2', 'maritalStatus_1', 'smartphone_3'], dtype=object)

#suspicious_features = ['employment_2', 'maritalStatus_1', 'employment_0',
#    'employment_1', 'employment_3', 'employment_4',
#    'employment_5', 'employment_6', 'employment_7', 'maritalStatus_0',
#    'maritalStatus_2', 'maritalStatus_3', 'maritalStatus_4',
#    'maritalStatus_5', 'maritalStatus_6']
#features_df = features_df.drop(suspicious_features, axis=1)

# split into the X and y vectors:
X, y = features_df.iloc[:,:-1].values, features_df.iloc[:, -1]
X_names = features_df.columns.values[:-1]

print features_df.columns
print X_names

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
lr.score(X_test_std, y_test) # is this right? affected by some of the vars?
# look at the coefficients.
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


















#def get_features_from_game_record(recordId, data)

# load a single game record (for now..)
recordtoget = data[data['recordId']=='5a0b4204-8a6c-430f-be93-c5aa2d6c9e33']
#game_record = mt.load_memory_results_json(filePaths, data.game_records_txt[0])
game_record = mt.load_memory_results_json(filePaths, recordtoget.game_records_txt[0])



# get just one game to play with:
game = game_record[0]
game.keys()



def dictstring_to_nums(dictstring):
    '''
    takes a string, like u'{216.33, 267.33}' or
    u'{{36, 199}, {114, 114}}', and converts to a
    list of floats.
    '''
    nums = re.findall('[\d\.]+', dictstring)
    nums = [float(num) for num in nums]
    return nums

def pull_features_from_memory_game(game):

    def rect_locations(game):
        rawrects = game['MemoryGameRecordTargetRects']
        for rect in rawrects:

        return centers, radii


    def memorydist(ts, rects):
        '''
        ts is a single touch sample from one memory game.
        '''
        hitloc = ts['MemoryGameTouchSampleLocation']
        hitloc = dictstring_to_nums(hitloc) # hitloc = [x, y]
        trueloc =
        return memdist

#    def mean_dist_corrects(game):
    touchsamples = game['MemoryGameRecordTouchSamples']
    for ts in touchsamples:
        if ts['MemoryGameTouchSampleIsCorrect'] == True:
            dist_from_center = memorydist(ts)


        pass
        return meandistcorrects, touchsamples


    # outputs:
    memory_features = {}
    memory_features['score'] = game['MemoryGameRecordGameScore']
    memory_features['meandistcorrects'] = mean_dist_corrects(game)
    pass
    return memory_features


def combine_memory_games_per_record(memory_features):
    pass


    return memory_features_combined


# represent the memory squares graphically:
rects = game['MemoryGameRecordTargetRects']
rects_as_nums = []
for rect in rects:
    r = re.findall('\d+', rect)
    r = [float(num) for num in r]
    rects_as_nums.append(r)
rects = rects_as_nums



#    cd /Users/matto/Dropbox/Insight/sage/


# convert_squares_to_patch
rect = rects[0]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
ax1.add_patch(patches.Rectangle((rect[0], rect[1]),   # (x,y)
        rect[2],          # width
        rect[3],          # height
    )
)





points = [[2, 1], [8, 1], [8, 4]]
polygon = plt.Polygon(points)
points = [[2, 4], [2, 8], [4, 6], [6, 8]]
line = plt.Polygon(points, closed=None, fill=None, edgecolor='r')



filePaths[df.ix[0, 'deviceMotion_walking_outbound.json.items’)]
filePaths[df.ix[0, 'deviceMotion_walking_outbound.json.items’)]







# visualize columns:
memory.hist()

# look at one entry:
memory[memory.recordId == '5a0b4204-8a6c-430f-be93-c5aa2d6c9e33']['game_numGames'] # picks numGames column for this recordId row



filePaths = load_memory_game_results()







# look at the records from one game:
# s = df['MemoryGameResults.json.MemoryGameGameRecords']


# first task: split into parkinsons' and non-parkinsons'...








#if __name__ == "__main__":
#    main()

    ####### float(tsloc.split(',')[0][1:])


