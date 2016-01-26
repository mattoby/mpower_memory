# memorytools
# this will be the package that holds the memory toolkit


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

###############################
## Pulling data from synapse ##
###############################

def login_synapse(username, password):
    '''
    Log into Synapse.

    For example,
    >>> login_synapse(bla, bla)
    5

    :param str username: the first number
    :param int b: the second number
    :returns: the syn structure logged in as user
    '''
    syn = synapseclient.Synapse()
    syn.login(username, password) # need to change this, security.
    return syn


def load_memory_table_from_synapse(syn):
    '''
    Get the memory table up and running:
    '''
    memorysyn = syn.tableQuery("select * from %s" % 'syn5511434')
    memory = memorysyn.asDataFrame()

    # rename columns: #!# clean this up later..
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

    # rename columns: #!# clean this up later..
    demographics.columns = ([u'recordId_demographic', u'healthCode', u'createdOn_demographic', u'appVersion_demographic', u'phoneInfo_demographic',
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


# create the whole environment for the memory data:
# (this is the last setup function, which folds in the others)
def create_memory_environment(synapse_user, synapse_pass):

    syn = login_synapse(synapse_user, synapse_pass)
    memory, memorysyn = load_memory_table_from_synapse(syn)
    filePaths = load_memory_game_results_from_synapse(syn, memorysyn)
    demographics, demosyn = load_demographics_table_from_synapse(syn)

    ## join dataframes:
    def has_parkinsons(data):
        hasdiagyear = ~np.isnan(data.diagYear)
        hasprofessionalDiagnosis = data.professionalDiagnosis == True
        hasParkinsons = hasdiagyear | hasprofessionalDiagnosis
        return hasParkinsons
    data = pd.merge(left=memory, right=demographics, how='inner', left_on='healthCode', right_on='healthCode')
    data['hasParkinsons'] = has_parkinsons(data)

    ## check dataset:
    assert len(data['recordId']) == len(data), 'Memory test record Ids should be unique, but they aren''t -- check the data structure.'

    return syn, memory, memorysyn, filePaths, demographics, demosyn, data


#####################
## Filter the data ##
#####################

def filter_data_for_popular_phones(data):
    '''
    only include the phones with a lot of records
    '''
    numuserscutoff = 1000
    phonegroups = data.groupby('phoneInfo').size()
    goodphones = phonegroups[phonegroups > numuserscutoff].index
    data = data[data.phoneInfo.isin(goodphones)]
    print "(phones are now filtered for only the most popular ones)"
    return data


#######################################
## Get features from the memory game ##
#######################################

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
    print game

    '''
    Main function to pull features out of memory game records.
    '''
    def rect_locations(game):
        '''
        returns x_ctr, y_ctr, radius for each patch in memory game
        (each row is a patch)
        '''
        rawrects = game['MemoryGameRecordTargetRects']
        rectlocations = []
        for rect in rawrects:
            rectloc = dictstring_to_nums(rect)
            x = rectloc[0] # bottom left x
            y = rectloc[1] # bottom left y
            dx = rectloc[2]/2.0 # dist to center, x-axis
            dy = rectloc[3]/2.0 # dist to center, y-axis
            assert dx == dy, 'game patches aren''t squares. height=%s, width=%s. check' % (dy, dx)
            radius = np.mean((dx,dy))
            x_ctr = x + dx
            y_ctr = y + dy
            rectlocations.append([x_ctr, y_ctr, radius])
        return rectlocations

    def memorydist(ts, rectlocations):
        '''
        outputs the distance from correct patch center to where player touched.
        ts is a single touch sample from one memory game.
        row of ts['..TargetIndex'] corresponds to row ind from rectlocations.
        '''
        hitlocraw = ts['MemoryGameTouchSampleLocation']
        hitloc = dictstring_to_nums(hitlocraw) # hitloc = [x, y]
        trueind = ts['MemoryGameTouchSampleTargetIndex'] # index from 0
        trueloc = rectlocations[trueind]
        x_true = trueloc[0]
        y_true = trueloc[1]
        x_hit = hitloc[0]
        y_hit = hitloc[1]
        memdist = np.sqrt((x_true - x_hit)**2 + (y_true - y_hit)**2)
        return memdist

    def memorydists(touchsamples, rectlocations):
        '''
        pulls distance information from touchsamples from 1 game
        (calls memorydist)
        '''
        memdists = []
        for ts in touchsamples:
            memdists.append(memorydist(ts, rectlocations))

        print 'memdists = %s' % memdists

        firstdist = memdists[0]
        meandist = np.mean(memdists) # does this make sense?

        return firstdist, meandist, memdists

    def memorytimes(touchsamples):
        '''
        takes in all touchsamples, and outputs the time delay for each
        '''
        touchtimes = []
        touchDtimes = []
        for ind, ts in enumerate(touchsamples):
            touchtimes.append(ts['MemoryGameTouchSampleTimestamp'])
            if ind == 0:
                touchDtimes.append(touchtimes[ind])
            else:
                touchDtimes.append(touchtimes[ind] - touchtimes[ind - 1])

        latency = touchDtimes[0] # the wait before first touch
        meanDt = np.mean(touchDtimes[1:]) # the mean wait between the next touches
        return latency, meanDt, touchDtimes #, touchtimes

    def memorysuccesses(touchsamples):
        '''
        returns success status of each touch sample
        '''
        successes = []
        for ts in touchsamples:
            successes.append(ts['MemoryGameTouchSampleIsCorrect'])
        successful = all(successes)
        return successful, successes

    # main function actions:

    # define rectangle location centers & touch samples:
    touchsamples = game['MemoryGameRecordTouchSamples']
    rectlocations = rect_locations(game)

    # find distances:
    firstdist, meandist, memdists = memorydists(touchsamples, rectlocations)
     # find times:
    latency, meanDt, touchDtimes = memorytimes(touchsamples)
    # find successes:
    successful, successes = memorysuccesses(touchsamples)
    # split dists into success & non-success categories:
#    print 'firstdist=%s' % firstdist
#    print 'meandist=%s' % meandist
#    print 'memdists=%s' % memdists
#    print 'latency=%s' % latency
#    print 'meanDt=%s' % meanDt
#    print 'touchDtimes=%s' % touchDtimes
#    print 'successes=%s' % successes
#    print 'successful=%s' % successful




#    if len(successes) > 1:
#        print successes
#        print game

#        assert successes[-2] == True, 'the second to last success is not true, it should always be!' # this assert is wrong.

    # game board size (larger = harder! analyze separately!):
    gamesize = game['MemoryGameRecordGameSize']
    # game score:
    gamescore = game['MemoryGameRecordGameScore']

    print 'gamesize=%s' % gamesize

    # group dists into success & non-success categories:
    successdists = [memdists[i] for i in range(len(memdists)) if successes[i]]
    unsuccessdists = [memdists[i] for i in range(len(memdists)) if successes[i]==False]

    # pack outputs:

    # distances, etc. in uncondensed form (1 ind per touch sample)
    memory_features_uncondensed = {}
    memory_features_uncondensed['memdists'] =memdists
    memory_features_uncondensed['successes'] = successes
    memory_features_uncondensed['memdists_successful'] = successdists
    memory_features_uncondensed['memdists_unsuccessful'] = unsuccessdists

    # distances, times, etc. condensed into single stats per game
    memory_features = {}
    memory_features['firstdist'] = firstdist
    memory_features['meandist'] = meandist
    memory_features['latency'] = latency
    memory_features['meanDt'] = meanDt
    memory_features['successful'] = successful
    memory_features['gamesize'] = gamesize
    memory_features['gamescore'] = gamescore
    memory_features['meansuccessfuldist'] = np.mean(successdists)
    memory_features['meanunsuccessfuldist'] = np.mean(unsuccessdists)
    memory_features['numsuccesses'] =  np.sum(np.array(successes))
    memory_features['numunsuccesses'] =  np.sum(~np.array(successes))
#    featurenames = memory_features.keys()
#    print memory_features
    return memory_features #, memory_features_uncondensed


def extract_games_from_memory_record(filePaths, data, memrecordId):
    '''
    pulls games out of a single record of the memory table
    reference by recordId from the memory table
    '''
    recordtoget = data[data['recordId']==memrecordId]
#    record_Id = data.game_records_txt[0]
#    game_record = load_memory_results_json(filePaths, record_Id)
    print recordtoget.game_records_txt
    record_Id = recordtoget.game_records_txt.values[0]
    print record_Id
    games_from_record = load_memory_results_json(filePaths, record_Id)
    return games_from_record


allowedgamesizes = np.array([4, 9, 16])

def group_games_by_sizes(games, allowedgamesizes):
    '''
    Group 'games' from record into groups, one per allowed gamesize
    This will be output as a dict, where keys are the gamesizes,
    and each value is a list of outputs of all games with that size.
    '''
    #allowedgamesizes = np.array([4, 9, 16])

    gamesizes = []
    for game in games:
        gamesizes.append(game['MemoryGameRecordGameSize'])

    assert set(gamesizes).issubset(set(allowedgamesizes)), 'not all of the gamesizes are accounted for. allowedgamesizes=%s, and gamesizes=%s. Add more to the allowed game sizes, or errorcheck.' % (allowedgamesizes, gamesizes)

    # create dict to hold games grouped by sizes:
    games_by_sizes = {}
    for allowedsize in allowedgamesizes:
        games_by_sizes[allowedsize] = []

    # group together games of the given gamesize:
#    games_by_sizes = [None]*len(allowedgamesizes)
    for ind, game in enumerate(games):
        gamesize = gamesizes[ind]
#        allowedind = np.where(allowedgamesizes == gamesize)[0][0]
        games_by_sizes[gamesize].append(game)

    return games_by_sizes #, allowedgamesizes


def average_features_from_memory_games(games):
    '''
    pulls features out of a set of games
    (i.e., from one record of the memory table)
    '''
    # add assert that game sizes are all the same!

    all_memory_features = {}
    for game in games:
        memory_features = pull_features_from_memory_game(game)
        for feature in memory_features:
#            print feature
            if all_memory_features.has_key(feature):
                all_memory_features[feature].append(memory_features[feature])
            else:
                all_memory_features[feature] = [memory_features[feature]]

    avg_memory_features = {}
    for feature in all_memory_features:
        avg_memory_features[feature] = np.mean(all_memory_features[feature])

    return avg_memory_features # , all_memory_features


def form_features_from_memory_record(filePaths, data, memrecordId, allowedgamesizes):
    '''
    This does the full pipeline for a single memory table record:
    splits them into game size groups, & determines averaged
    features for each group
    '''
        # pull out games:
    games = extract_games_from_memory_record(filePaths, data, memrecordId)
    games = filter_out_broken_games(games)
    games_by_sizes = group_games_by_sizes(games, allowedgamesizes)

        # split them up by game sizes:
    avg_features_by_sizes = {}
    for gamesize in games_by_sizes:
        games = games_by_sizes[gamesize]
        if len(games) > 0:
            avg_memory_features = average_features_from_memory_games(games)
            avg_features_by_sizes[gamesize] = avg_memory_features
#            else
#                memory_features_by_sizes[gamesize] = []
    return avg_features_by_sizes

def filter_out_broken_games(games):
    '''
    remove games that are in some way messed up (eg., no touch records)
    '''
    # Tag games as broken:
    brokengametag = [False]*len(games)
    for n, game in enumerate(games):
        # no touch samples:
        if len(game['MemoryGameRecordTouchSamples']) == 0:
            brokengametag[n] = True
        # other criteria:?

    # remove broken ones:
    games = [games[i] for i in range(len(games)) if brokengametag[i]==False]

    return games



###############
## Old/other ##
###############

## represent the memory squares graphically:
#rects = game['MemoryGameRecordTargetRects']
#rects_as_nums = []
#for rect in rects:
#    r = re.findall('\d+', rect)
#    r = [float(num) for num in r]
#    rects_as_nums.append(r)
#rects = rects_as_nums
#
## convert_squares_to_patch
#rect = rects[0]
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111, aspect='equal')
#ax1.add_patch(patches.Rectangle((rect[0], rect[1]),   # (x,y)
#        rect[2],          # width
#        rect[3],          # height
#    )
#)
#
#points = [[2, 1], [8, 1], [8, 4]]
#polygon = plt.Polygon(points)
#points = [[2, 4], [2, 8], [4, 6], [6, 8]]
#line = plt.Polygon(points, closed=None, fill=None, edgecolor='r')










