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
import numpy as np
from sklearn.utils.validation import check_consistent_length, _num_samples

features = ['game_score','game_numGames'] # cannot have just 1 feature
X = data[features]

Y = data['hasParkinsons'].astype('int')

logr = linear_model.LogisticRegression()

logr.fit( X , Y )


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


# features_df.isnull().sum() # looks at the nulls in each col
#education - 8 nans
#employment - 19 nans
#gender - 5 nans
#lastSmoked - >5000 nans
#maritalStatus - 18 nans
#phoneUsage - 6 nans
#smartphone - 5 nans
# age - 15 nans


features_df = data[["game_score","phoneInfo",
    "game_numFails","age","brainStim","education",
    "employment","gender","maritalStatus",
    "phoneUsage", "smartphone", "hasParkinsons"]]

names_of_columns_to_transform = ["phoneInfo",
    "brainStim","education","employment",
    "gender","maritalStatus",
    "phoneUsage", "smartphone"]
# diagYear... onsetYear

for column in names_of_columns_to_transform:
    features_df = transform_feature( features_df, column )

print( features_df.head() )

# drop nas:
# notnanrows = np.where(features_df['age'].notnull())[0]
features_df = features_df.dropna()

y = features_df['hasParkinsons'].astype('int').tolist()
features_df = features_df.drop('hasParkinsons',1)
X = features_df.as_matrix()


clf1 = linear_model.LogisticRegression()
score1 = sklearn.cross_validation.cross_val_score( clf1, X, y )
print( score1 )

import sklearn.tree
import sklearn.ensemble
​

clf2 = sklearn.tree.DecisionTreeClassifier()
score2 = sklearn.cross_validation.cross_val_score( clf2, X, y )
print( score2 )
​
clf3 = sklearn.ensemble.RandomForestClassifier()
score3 = sklearn.cross_validation.cross_val_score( clf3, X, y )
print( score3 )









#def get_features_from_game_record(recordId, data)

# load a single game record (for now..)
recordtoget = data[data['recordId']=='5a0b4204-8a6c-430f-be93-c5aa2d6c9e33']
#game_record = mt.load_memory_results_json(filePaths, data.game_records_txt[0])
game_record = mt.load_memory_results_json(filePaths, recordtoget.game_records_txt[0])



# get just one game to play with:
game = game_record[0]
game.keys()

# represent the memory squares graphically:
rects = game['MemoryGameRecordTargetRects']
rects_as_nums = []
for rect in rects:
    r = re.findall('\d+', rect)
    r = [int(num) for num in r]
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


