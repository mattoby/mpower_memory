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
%matplotlib inline
# from pandas import DataFrame, Series


%pylab # if running in ipython... (so i can see plots)


# log in
def login_synapse(username, password):
    syn = synapseclient.Synapse()
    syn.login(username, password) # need to change this, security.
    return syn

# get the memory table up and running:
def load_memory_table_from_synapse(syn):
    memorysyn = syn.tableQuery("select * from %s" % 'syn5511434')
    memory = memorysyn.asDataFrame()

    # rename columns:
    memory.columns = [u'recordId', u'healthCode', u'createdOn',
        u'appVersion', u'phoneInfo', u'game_score',
        u'game_numGames', u'game_numFails', u'game_startDate',
        u'game_endDate', u'game_records', u'medTimepoint']

    # remove rows with null in the game_records column:
    memory = memory.dropna(subset=["game_records"])

    # Convert 'game_records' into a text key that matches filePaths:
    for rec in memory.game_records:
        assert (len(str(rec)) == 9), "records will not be parsed right since they are not the correct length - check %s" % rec
    memory['game_records_txt'] = memory.game_records.apply(lambda r: str(r)[:7])

    # finish:
    return memory, memorysyn

# get the demographics table up and running:
def load_demographics_table_from_synapse(syn):
    demosyn = syn.tableQuery("select * from %s" % 'syn5511429')
    demographics = demosyn.asDataFrame()

    # rename columns:
    demographics.columns = ([u'recordId', u'healthCode', u'createdOn', u'appVersion', u'phoneInfo',
       u'age', u'isCaretaker', u'brainStim', u'diagYear',
       u'education', u'employment', u'gender', u'healthHistory',
       u'healthcareProvider', u'homeUsage', u'lastSmoked', u'maritalStatus',
       u'medicalUsage', u'medicalUsageYesterday', u'medicationStartYear',
       u'onsetYear', u'packsPerDay', u'pastParticipation', u'phoneUsage',
       u'professionalDiagnosis', u'race', u'smartphone', u'smoked',
       u'surgery', u'videoUsage', u'yearsSmoking'])

    # finish:
    return demographics, demosyn


# get the json files (slow, only load from scratch once):
def load_memory_game_results_from_synapse(syn, memorysyn, fromScratch = False):
    if fromScratch:
        filePaths = syn.downloadTableColumns(memorysyn, u'MemoryGameResults.json.MemoryGameGameRecords')
        pickle.dump( filePaths, open( "filePaths_for_memory.p", "wb" ) )
    else:
        filePaths = pickle.load( open( "filePaths_for_memory.p", "rb" ) )
    return filePaths


# load in the json data for a memory test:
def load_memory_results_json(filePaths, game_record_txt):
#    with open(filePaths[u'5732386']) as data_file:
    with open(filePaths[game_record_txt]) as data_file:
        game_record = json.load(data_file)
    return game_record




## load up data:
syn = login_synapse(os.environ['SYNAPSE_USER'], os.environ['SYNAPSE_PASS'])
memory, memorysyn = load_memory_table_from_synapse(syn)
filePaths = load_memory_game_results_from_synapse(syn, memorysyn)
demographics, demosyn = load_demographics_table_from_synapse(syn)

## join dataframes:
data = pd.merge(left=memory, right=demographics, how='inner', left_on='healthCode', right_on='healthCode')

# load a single game record (for now..)
game_record = load_memory_results_json(filePaths, memory.game_records_txt[0])

# get just one game to play with:
game = game_record[0]

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


