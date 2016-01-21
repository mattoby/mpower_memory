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
%pylab # if running in ipython... (so i can see plots)

## load up data:
syn = mt.login_synapse(os.environ['SYNAPSE_USER'], os.environ['SYNAPSE_PASS'])
memory, memorysyn = mt.load_memory_table_from_synapse(syn)
filePaths = mt.load_memory_game_results_from_synapse(syn, memorysyn)
demographics, demosyn = mt.load_demographics_table_from_synapse(syn)

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


