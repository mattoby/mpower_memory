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
import sys
from numpy import nan
import seaborn as sns
import datetime
from scipy.stats import ttest_ind
from scipy.stats import ranksums
from contextlib import contextmanager


from matplotlib.colors import ListedColormap

# sklearn imports:
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#####################################
## fixed variables for memorytools ##
#####################################


agecutoff = 45 # age below which there are almost no Parkinsons patients

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
    syn.login(username, password, rememberMe=True)
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


loadSynapseRecordsFromScratch = False
# get the json files (slow, only load from scratch once):
def load_memory_game_results_from_synapse(syn, memorysyn, fromScratch = loadSynapseRecordsFromScratch):
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


def join_memory_and_demographics_dfs(memory, demographics):
    '''
    Join the memory and demographics dataframes into new dataframe,
    'data'
    '''
    data = pd.merge(left=memory, right=demographics, how='inner', left_on='healthCode', right_on='healthCode')
    return data


# create the whole environment for the memory data:
# (this is the last setup function, which folds in the others)
def create_memory_environment(synapse_user, synapse_pass):

    ## start everything up:
    syn = login_synapse(synapse_user, synapse_pass)
    memory, memorysyn = load_memory_table_from_synapse(syn)
    filePaths = load_memory_game_results_from_synapse(syn, memorysyn)
    demographics, demosyn = load_demographics_table_from_synapse(syn)

    ## join dataframes:
    data = join_memory_and_demographics_dfs(memory, demographics)

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

def add_composite_features_to_data(data):
    '''
    Define special composite columns in the data, which are calculated from other raw columns.
    '''
    def has_parkinsons(data):
        hasdiagyear = ~np.isnan(data.diagYear)
        hasprofessionalDiagnosis = data.professionalDiagnosis == True
        hasParkinsons = hasdiagyear | hasprofessionalDiagnosis
        return hasParkinsons

    def played_game4(data):
        '''
        Did they play the 2x2 game in that session?
        '''
        playedgame4 = data['4_gamesize'].notnull()
        return playedgame4

    curryear = datetime.datetime.now().year
    def nyears_parkinsons(data, OutlierCutoff = 50, curryear=curryear):
        '''
        Number of years that somebody has had parkinsons
        Outliers (default = those above 50 years) are lowered to the next highest val (they are assumed to be incorrectly entered in the app)
        '''
        nyears = np.array([curryear - y for y in data['onsetYear']])
        # fix outliers:
        newmax = nyears[nyears < OutlierCutoff].max()
        nyears[nyears > OutlierCutoff] = newmax
        return nyears

    def nyears_on_meds(data, OutlierCutoff = 50, curryear=curryear):
        '''
        Number of years that a patient has been on parkinsons meds
        (assumes that the patients have been taking meds continuously since they started)
        Note: some of the patients put in the year '0' or '15' for when they started. these are converted to nulls (they should be years AD). The outlier cutoff is 50 years (i.e., 50 years back from today's date, which would be, e.g., 1966, rather than the year 0, e.g.)
        '''
        nyears = np.array([curryear - y for y in data['medicationStartYear']])
        # fix outliers, by making them nan:
        nyears[nyears > OutlierCutoff] = nan
        return nyears


    ## add extra features:
    data['hasParkinsons'] = has_parkinsons(data)
    data['played_game4'] = played_game4(data)
    data['nyearsParkinsons'] = nyears_parkinsons(data)
    data['nyearsOnMeds'] = nyears_on_meds(data)
    data['nyearsOffMeds'] = data['nyearsParkinsons'] - data['nyearsOnMeds']
    print 'Note that nyearsOffMeds = nyearsParkinsons - nyearsOnMeds'

    ## idiot check outputs:
    assert sum(data['nyearsParkinsons']<data['nyearsOnMeds']) == 0, 'Some of the nyearsOnMeds records are higher than the nyearsParkinsons records - there is a problem here'

    return data


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
#    print game

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

#        print 'memdists = %s' % memdists

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
        successes = record of touch successes [True, False, False]
        successful = if game was successful (success in all touches)
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

#    print 'gamesize=%s' % gamesize

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
#    print recordtoget.game_records_txt
    record_Id = recordtoget.game_records_txt.values[0]
    print 'Extracting games record from record_Id = %s' % memrecordId
    games_from_record = load_memory_results_json(filePaths, record_Id)
    return games_from_record


allowedgamesizes = np.array([4, 9, 16])

def group_games_by_sizes(games, allowedgamesizes=allowedgamesizes):
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

#    print 'all memory features: %s' % all_memory_features
    avg_memory_features = {}
    for feature in all_memory_features:
        # pull out the values for that feature:
        vals = np.array(all_memory_features[feature])
        # remove nan values:
        vals = vals[~np.isnan(vals)]
        # average non-nan values:
        avg_memory_features[feature] = np.mean(vals)
#        avg_memory_features[feature] = np.mean(all_memory_features[feature])

    return avg_memory_features # , all_memory_features


def form_features_from_memory_record(filePaths, data, memrecordId, allowedgamesizes=allowedgamesizes):
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
        currgames = games_by_sizes[gamesize]
        if len(currgames) > 0:
            avg_memory_features = average_features_from_memory_games(currgames)
            avg_features_by_sizes[gamesize] = avg_memory_features
#            else
#                memory_features_by_sizes[gamesize] = []


    return avg_features_by_sizes, games, games_by_sizes




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


def add_memory_game_features_to_data(filePaths, data, allowedgamesizes=allowedgamesizes, fromFile=True, outFileName='memory_data_with_features.p', toSave=False):
    '''
    Master function that goes through records in 'data' (the merged memory & demographics dataframe), pulls out features from the memory game
    Note: if fromScratch=False, then the data & filepaths are ignored, and pre-processed data is imported from savefilename. Otherwise, it is saved

    '''
    if not(fromFile):
        data['gamesdata'] = 0
#        print '\n\n\n\n'
#        print data['gamesdata']
#        print '\n\n\n\n'

        for memrecordId in data['recordId']:
            rowidx = data[data['recordId']==memrecordId].index.tolist()
#            rowidx = rowidx[0]
            print 'Adding features to row: %s' % rowidx#[0]

            avg_features_by_sizes, games, games_by_sizes = form_features_from_memory_record(filePaths, data, memrecordId, allowedgamesizes)

            # add games & games_by_sizes to data as new column:
#            print '\n\ngames = %s\n\n\n\n\n' % games
#            print '\n\nrowidx = %s\n\n\n\n\n' % rowidx
#            print '\n\ngames_by_sizes = %s\n\n\n\n\n' % games_by_sizes
            #gamesdata = {'games':games, 'games_by_sizes':games_by_sizes}

            gamesdata = {'a':games_by_sizes}

#            data.set_value(rowidx, 'gamesdata', gamesdata)
            data.set_value(rowidx, 'gamesdata', {'games_by_sizes':games_by_sizes})
#            data.set_value(rowidx, 'games_by_sizes', games_by_sizes)

            # put features into data structure:
            for gamesize in avg_features_by_sizes:
#                print 'avg_features_by_sizes = %s' % avg_features_by_sizes
                features_within_gamesize = avg_features_by_sizes[gamesize]

                for feature in features_within_gamesize:
                    colname = '%s_%s' % (gamesize, feature)
                    featureval = features_within_gamesize[feature]

                    data.set_value(rowidx, colname, featureval)
        print 'memory features extracted from inputted data'
    else:
        data = pickle.load( open( outFileName, "rb" ) )
        print 'memory features loaded from file: %s (input data was ignored)' % outFileName
    if toSave:
        pickle.dump( data, open( outFileName, "wb" ) )
        print 'memory features saved to file: %s' % outFileName

    print 'Warning: need to deal with case where meansuccessfuldist > meanunsuccessfuldist, e.g., with record 7944 (1st 16-box game)'
    return data


def extract_health_history_words(data):
    '''
    Extract all words from the healthHistory field (in case useful)
    Might be useful... text analysis...
    '''


def define_feature_categories():
    fcats = {}

    fcats['demographic'] = \
        ['age',
        'gender',
        'education']

    fcats['game'] = \
        ['game_numFails',
        'game_score',
        'game_numGames',
        '9_numsuccesses',
        '9_numunsuccesses',
        '9_meandist',
        '9_successful',
        '9_gamescore',
        '9_latency',
        '9_firstdist',
        '9_meanDt',
        '9_meansuccessfuldist',
        '16_firstdist',
        '16_meandist',
        '16_numsuccesses',
        '16_gamescore',
        '16_latency',
        '16_numunsuccesses',
        '16_successful',
        '16_meanDt',
        '16_meansuccessfuldist',
        'played_game4']

    fcats['phone'] = \
        ['phoneInfo',
        'smartphone']

    fcats['output'] = \
        ['hasParkinsons',
        'medTimepoint',
        'brainStim',
        'surgery',
        'nyearsOnMeds',
        'nyearsOffMeds',
        'nyearsParkinsons']
    print 'Note that nyearsOffMeds = nyearsParkinsons - nyearsOnMeds'

    fcats['time'] = \
        ['game_endDate',
        'createdOn',
        'game_startDate']

    fcats['person'] = ['healthCode']
    return fcats

def feature_names(features):
    '''
    Names for all the features (for display)
    returns a list of feature names corresponding to the features input
    '''
    fnames={
        'age':                  'age',
        'gender':               'gender',
        'education':            'education',
        'game_numFails':        '# failed games',
        'game_score':           'memory score (overall)',
        'game_numGames':        '# games played',
        '9_numsuccesses':       '# successful taps (3x3)',
        '9_numunsuccesses':     '# unsuccessful taps (3x3)',
        '9_meandist':           'mean distance (3x3)',
        '9_successful':         '% games won (3x3)',
        '9_gamescore':          'memory score (3x3)',
        '9_latency':            'reaction time (3x3)',
        '9_firstdist':          'first tap distance (3x3)',
        '9_meanDt':             'mean time between taps (3x3)',
        '9_meansuccessfuldist': 'mean correct tap distance (3x3)',
        '16_firstdist':         'first tap distance (4x4)',
        '16_meandist':          'mean tap distance (4x4)',
        '16_numsuccesses':      '# successful taps (4x4)',
        '16_gamescore':         'memory score (4x4)',
        '16_latency':           'reaction time (4x4)',
        '16_numunsuccesses':    '# unsuccessful taps (4x4)',
        '16_successful':        '% games won (4x4)',
        '16_meanDt':            'mean time between taps (4x4)',
        '16_meansuccessfuldist':'mean correct tap distance (4x4)',
        'played_game4':         'played 2x2 game',
        'phoneInfo':            'phone screen size',
        'smartphone':           'ease of phone use',
        'hasParkinsons':        'has Parkinson''s?',
        'medTimepoint':         'just took meds?',
        'brainStim':            'had brain stimulation?',
        'surgery':              'had surgery?',
        'nyearsOnMeds':         '# years on medication',
        'nyearsOffMeds':        '# years nonmedicated',
        'nyearsParkinsons':     '# years of Parkinson''s',
        'game_endDate':         'game date'
        }
    fnames_out = np.array([fnames[f] for f in features])
    return fnames_out


#################################
## Machine Learning, data prep ##
#################################

# maps of different features to ordinal values
feature_ordinal_maps = {
    'education_code':
        {'Some high school':2,
        'High School Diploma/GED':4,
        '2-year college degree':6,
        'Some college':6,
        '4-year college degree':8,
        'Some graduate school':10,
        "Master's Degree":10,
        'Doctoral Degree':13},
    ### Smartphone by difficulty description:
    'smartphone_code':
        {'Very easy':1,
        'Easy':2,
        'Neither easy nor difficult':3,
        'Difficult':4,
        'Very Difficult':5},
    ### Gender (binary ordinates):
    'gender_code':
        {'Male':1,
        'Female':0},
    ### Phone Usage (what does this mean?):
    'phoneUsage_code':
        {'false':0,
        'Not sure':1,
        'true':2},
    ### Phone Info (the phone used) (encoded as screen size):
    'phoneInfo_code':
        {'iPhone 5s (GSM)':4.0,
        'iPhone 6':4.7,
        'iPhone 6 Plus':5.5},
    ### medTimepoint (note, this output is only of whether they are on meds)
    'medTimepoint_code':
        {"I don't take Parkinson medications":nan,
        "Immediately before Parkinson medication":0.0,
        "Another time":nan,
        "Just after Parkinson medication (at your best)":1.0 }
    }

def convert_features_to_numbers(features_df, feature_ordinal_maps=feature_ordinal_maps, featuresToConvert =[]):
    '''
    Prep step of particular features, which are categorical (but ordered) and should be converted to ordinal or cardinal #'s - this converts them to numbers for import to machine learning model.
    Will not convert columns that are already completely non-strings
    Should be fixed to deal better with nans.
    if featuresToConvert is empty, then it will convert all
    '''
    df = pd.DataFrame.copy(features_df)

    def ordinate_categorical_col(df, column, code):
        '''
        convert a categorical (but ordered) column to #'s, based on a manually determined conversion code.
        '''
        def assign_code(code, colval):
#            print 'COLVAL IS!!!!!!!!! %s' % colval
#            print 'code is!!!! %s' % code
            if pd.isnull(colval):
                return nan
            else:

                return code[colval]

#        assert set(df[column].unique())==set(education_code.keys()), 'Need to make new code maps - it doesn''t match the codes from the data'
        # this assert should be done!!! problem with nans!

        # reset smartphone values to difficulty for user:
#        df[column] = [code[df[column][i]] for i in df.index]
        df[column] = [assign_code(code, df[column][i]) for i in df.index]

        return df

    ### Define maps of categories to ordinal values:
    education_code =    feature_ordinal_maps['education_code']
    smartphone_code =   feature_ordinal_maps['smartphone_code']
    gender_code =       feature_ordinal_maps['gender_code']
    phoneUsage_code =   feature_ordinal_maps['phoneUsage_code']
    phoneInfo_code =    feature_ordinal_maps['phoneInfo_code']
    medTimepoint_code = feature_ordinal_maps['medTimepoint_code']

    fcodes = {'smartphone':smartphone_code,
    'education':education_code,
    'gender':gender_code,
    'phoneUsage':phoneUsage_code,
    'phoneInfo':phoneInfo_code,
    'medTimepoint':medTimepoint_code}

    # determine which features to ordinate:
    # if featuresToConvert is [], then this will pick all
    if len(featuresToConvert) > 0:
        fcodesgood = {}
        for feature in featuresToConvert:
            fcodesgood[feature] = fcodes[feature]
            fcodes = fcodesgood

    featureschanged = []
    for feature in fcodes:
#        print 'feature = %s' % feature
        if feature in df:
            # check that the feature hasn't already been changed:
            # (this tests if there are strings still in the column)
            hasStrings = check_for_string_in_dfcol(df, feature)
            if hasStrings:
                # print 'featuretochange = %s' % feature
                df = ordinate_categorical_col(df, feature, fcodes[feature])
                featureschanged.append(feature)

#    df = ordinate_categorical_col(df, 'smartphone', smartphone_code)
#    df = ordinate_categorical_col(df, 'education', education_code)
#    df = ordinate_categorical_col(df, 'gender', gender_code)
#    df = ordinate_categorical_col(df, 'phoneUsage', phoneUsage_code)
#    df = ordinate_categorical_col(df, 'phoneInfo', phoneInfo_code)
    print 'Features converted to numbers:'
    print featureschanged #smartphone, education, gender, phoneUsage, phoneInfo'

    # convert boolean columns to ints (because of bug? in pandas):
    boolcols_to_convert_to_int = ['brainStim']
    for col in boolcols_to_convert_to_int:
        if col in df.columns:
            df = convert_boolean_col_to_int(df, col)
            print '%s converted to int' % col
    return df


def split_off_label_variable(features_df, labelcol):
    '''
    Split a column off of the features_df dataframe, and convert it to be used as the label array in sklearn
    Check similarity to move_col_to_end_of_df (should be combined)
    '''

    y_col = features_df[labelcol]
    features_df = features_df.drop(labelcol, axis=1)

    return features_df, y_col

def convert_features_df_to_X_and_y_for_machinelearning(features_df, labelcol):
    # split off the label variable:
    features_df, y_col = split_off_label_variable(features_df, labelcol)
    # split into the X and y vectors:
    X = features_df.iloc[:,:].values
    y = y_col.values
    X_names = features_df.columns.values
    y_name = y_col.name
    # # old way: X, y = features_df.iloc[:,:-1].values, features_df.iloc[:, -1]
    return X, y, X_names, y_name


def prep_memory_features_for_machine_learning(data, features, labelcol, convert_features_to_nums=True, dropnas=True, toStandardScale=False):
    '''
        Uses other ML prep functions to get memory features in and ready for machine learning.
        This takes in the memory data dataframe, the features of interest, and which feature should be the label column (i.e., what's being predicted), and formats all of this correctly for inputting into sklearn.
    '''

    ##################### Preprocess data for machine learning:
    # define features (include the label feature here:
#    features = ['game_score', 'age', 'hasParkinsons']
#    print features
    features_df = data[features]
    if convert_features_to_nums:
        features_df = convert_features_to_numbers(features_df)
    #features_df = move_col_to_end_of_df(features_df, 'hasParkinsons')
    features_df = move_col_to_end_of_df(features_df, labelcol)
    # do more processing here, in case of features with lots of nas?

    # drop na rows:
    if dropnas:
        features_df = features_df.dropna()
        print 'na rows have been dropped (if there were any)'

    # convert to matrices for machine learning:
    #labelcol = 'hasParkinsons'
    X, y, X_names, y_name = convert_features_df_to_X_and_y_for_machinelearning(features_df, labelcol)

    ##################### Set features up for machine learning:

    # split for cross validation:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 8)

    y_combined = np.hstack((y_train, y_test))

    # scale features:
    if toStandardScale:
        stdsc = StandardScaler()
        stdsc.fit(X_train)
        X_train_std = stdsc.fit_transform(X_train)
        X_test_std = stdsc.transform(X_test)
        X_combined_std = np.vstack((X_train_std, X_test_std))

        return features_df, X, y, X_names, y_name, X_train, X_test, y_train, y_test, stdsc, X_train_std, X_test_std, X_combined_std, y_combined

    else:
        return features_df, X, y, X_names, y_name, X_train, X_test, y_train, y_test



######################
## Machine learning ##
######################



def build_ML_model_age_corrected_and_samplebalanced(data, features, labelcol='hasParkinsons', toPlot=[1,1,1,1,1,1], toPrint=False, MLexcludecols=[]):
    '''
    Does age correction & sample balancing, then runs random forest ML

    This function is ugly, needs to be cleaned up & generalized
    features must include the labelcol

    Might need to build an option later that does not drop nas on all columns.. for now it does.

    MLexcludecols lists columns that should not go into the ML
    For example, if the labelcol should be
    '''

    # define the columns to sample balance & resample on:
    distcol = 'age'
    splitcol = 'hasParkinsons'
    nbins = 10
    nResamples = 600

    # build features dataframe:
    fdf = data[features]
    fdf = convert_features_to_numbers(fdf)

    # drop nas:
    len1 = len(fdf)
    fdf = fdf.dropna()
    len2 = len(fdf)
    print 'dropped %s rows to remove all nas from data' % (len1 - len2)

    # resample non-Park to same age distribution as Parkinsons:
    splitVal_resample = False
    splitVal_guide = True

    df_resampled, df_guide, df_resample = resample_to_match_distribution(fdf, distcol, splitcol, splitVal_resample, splitVal_guide, nbins, nResamples)
    df_resampled_np = df_resampled
    df_Parkinsons = df_guide
    df_np = df_resample

    # test pval first set:
    a = df_resampled[distcol].dropna().values
    b = df_guide[distcol].dropna().values
    p1 = ranksums(a, b)

    # resample Park to the resampled non-Park for sample balancing:
    fdf2 = df_resampled_np.append(df_Parkinsons)

    splitVal_resample = True
    splitVal_guide = False
    df_resampled, df_guide, df_resample = resample_to_match_distribution(fdf2, distcol, splitcol, splitVal_resample, splitVal_guide, nbins, nResamples)
    df_resampled_Park = df_resampled

    # test pval 2nd set:
    a = df_resampled_np[distcol].dropna().values
    b = df_resampled_Park[distcol].dropna().values
    p2 = ranksums(a, b)



    ### Redo machine learning with these sets:
    df = df_resampled_np.append(df_resampled_Park)

    # remove cols to exclude from ML (but that were needed for processing)
    if len(MLexcludecols) > 0:
        for col in MLexcludecols:
            df = df.drop(col, axis=1)
            features.remove(col)

    ######### Machine learning #########

    features_df, X, y, X_names, y_name, X_train, X_test, y_train, y_test = prep_memory_features_for_machine_learning(df, features, labelcol, convert_features_to_nums=False, toStandardScale=False)

    # create model:
    mod = RandomForestClassifier(n_estimators=100)
    #lr = linear_model.LogisticRegression(penalty='l1', C=0.1) # with regularization
    mod.fit(X_train, y_train)

    # Probabilities predicted for test set to be in + class:
    y_pred_proba = mod.predict_proba(X_test)[:,1]

    #  Confusion matrix:
    y_pred = mod.predict(X_test)
#    sklearn.metrics.roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # accuracies:
#    len()
    train_acc = mod.score(X_train, y_train)
    test_acc = mod.score(X_test, y_test)
    rand_acc = (float(sum(y))/len(y))
    precision = sklearn.metrics.precision_score(y_true=y_test, y_pred=y_pred)
    recall = sklearn.metrics.recall_score(y_true=y_test, y_pred=y_pred)
    F1 = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred)

    ######### Plotting & outputs #########

    if toPlot[0] == 1:

        # plot zeroth set:
        plt.figure()
        sns.distplot(df_Parkinsons[distcol].dropna(), label='Parkinsons')
        sns.distplot(df_np[distcol].dropna(), label='non Parkinsons')
        plt.legend(loc=2)
        plt.show()

    if toPlot[1] == 1:
        # plot first set:
        plt.figure()
        sns.distplot(df_Parkinsons[distcol].dropna(), label='Parkinsons')
        sns.distplot(df_np[distcol].dropna(), label='non Parkinsons')
        sns.distplot(df_resampled_np[distcol].dropna(), label='non Parkinsons, resampled')
        plt.legend(loc=2)
        plt.show()

    if toPlot[2] == 1:
    # plot second set:
        plt.figure()
        sns.distplot(df_Parkinsons[distcol].dropna(), label='Parkinsons')
        sns.distplot(df_np[distcol].dropna(), label='non Parkinsons')
        sns.distplot(df_resampled_np[distcol].dropna(), label='non Parkinsons, resampled')
        sns.distplot(df_resampled_Park[distcol].dropna(), label='Parkinsons, resampled')
        plt.legend(loc=2)
        plt.show()

    if toPlot[3] == 1:
        render_confusion_matrix(y_test, y_pred)

    if toPlot[4] == 1:
        plot_feature_importances_randforest(mod, X_names)

    if toPlot[5] == 1:
        plot_roc_curve(y_test, y_pred_proba)

    if toPrint == True:
        # test pvals 1st and 2nd set:
        print '\n'
        print 'ranksum pval for dist. resampling = %s' % p1[1]
        print 'ranksum pval for sample balanced = %s' % p2[1]
        print '\n'
        print 'num actual positives = %s' % sum(y)
        print 'num actual negatives = %s' % (len(y) - sum(y))
        print '\n'
        print '###### performance #######'
        print 'precision:', precision
        print 'recall:', recall
        print 'F1:', F1
        print 'training accuracy:', train_acc
        print 'test accuracy:', test_acc
        print 'random accuracy would be %s' % rand_acc
        print '##########################'
        print '\n'

        # feature importances:
#        print 'feature importances:'
#        S = pd.Series(mod.feature_importances_, index=X_names, name="feature importances")
#        print S.sort_values()

    return mod, features_df, X, y, X_names, y_name, X_train, X_test, y_train, y_test, train_acc, test_acc, rand_acc, y_pred, y_pred_proba, fdf





sampleBalanceDefaultParams = {
    'distcol':'age',
    'splitcol':'hasParkinsons',
    'nbins':10,
    'nResamples':600,
    'splitVal_resample':False,
    'splitVal_guide':True,
    }

def build_ML_model(data, features, labelcol='hasParkinsons', toPlot=[0,0,0], toPrint=True, MLexcludecols=[], modelType ='randomforest', featureToMean=[], sampleBalance=False, sampleBalanceParams=sampleBalanceDefaultParams):
    '''
    This will run a classifier model on the parkinsons dataframe.
    Ugly to have this and the age corrected version. Need to reconcile them.
    For classification models. to do regression, use build_ML_regression
    Should combine these later too..
    To take 1 sample of each patient:

    grouped = data.groupby('healthCode')
    datasamp = grouped.apply(lambda x: x.sample(n=1))
    # remove young patients:
    datasamp = datasamp[datasamp['age']>50]

    To take mean of each patient:

    '''

    # build features dataframe:
    fdf = data[features]

    fdf = convert_features_to_numbers(fdf)

    # drop nas:
    len1 = len(fdf)
    fdf = fdf.dropna()
    len2 = len(fdf)
    print 'dropped %s rows to remove all nas from data' % (len1 - len2)

    # optionally boil each patient down to his mean:
    # (note, will also exclude this column & turn it into the index!)
    if len(featureToMean) > 0:
        fdf = groupby_col_and_avg_other_cols(fdf, featureToMean)
        features.remove(featureToMean[0])

        print '# positive in labelcol: ', fdf[labelcol].sum()
        print '# total in labelcol: ', len(fdf[labelcol])

    # optionally resample after boiling down to mean:
    if sampleBalance == True:
        # define the columns to sample balance on:
        distcol = sampleBalanceParams['distcol'] # 'age'
        splitcol = sampleBalanceParams['splitcol'] # 'hasParkinsons'
        nbins = sampleBalanceParams['nbins'] # 10
        nResamples = sampleBalanceParams['nResamples'] #600
        splitVal_resample = sampleBalanceParams['splitVal_resample'] # False
        splitVal_guide = sampleBalanceParams['splitVal_guide'] # True
        # resample non-Park to same age distribution as Parkinsons:
        df_resampled, df_guide, df_resample = resample_to_match_distribution(fdf, distcol, splitcol, splitVal_resample, splitVal_guide, nbins, nResamples)

        # test pval (we want this to not be significant):
        a = df_resampled[distcol].dropna().values
        b = df_guide[distcol].dropna().values
        p1 = ranksums(a, b)
        print 'pval for resampling (want nonsignificant): ', p1[1]

        # combine back the datasets:
        fdf = df_resampled.append(df_guide)

    # remove cols to exclude from ML (but that were needed for processing)
    if len(MLexcludecols) > 0:
        for col in MLexcludecols:
            fdf = fdf.drop(col, axis=1)
            features.remove(col)

    # prep feature matrix for machine learning:
    features_df, X, y, X_names, y_name, X_train, X_test, y_train, y_test = prep_memory_features_for_machine_learning(fdf, features, labelcol, convert_features_to_nums=False, toStandardScale=False)

    ######### Machine learning #########
    if modelType == 'randomforest':
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
#        print importances
    elif modelType == 'logisticregression':
#        model = linear_model.LogisticRegression(penalty='l1', C=0.1)
        model = linear_model.LogisticRegression(penalty='l1', C=1000)
        model.fit(X_train, y_train)
        importances = np.array(model.coef_[0])


#        print importances

    # Probabilities predicted for test set to be in + class:
    y_pred_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)


    # Accuracies:
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    rand_acc = (float(sum(y))/len(y))
    precision = sklearn.metrics.precision_score(y_true=y_test, y_pred=y_pred)
    recall = sklearn.metrics.recall_score(y_true=y_test, y_pred=y_pred)
    F1 = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred)

    ######### Plotting & outputs #########

    if toPlot[0] == 1:
        render_confusion_matrix(y_test, y_pred)

    if toPlot[1] == 1:
#        plot_feature_importances_randforest(model, X_names)
        plot_feature_importances(X_names, importances)

    if toPlot[2] == 1:
        plot_roc_curve(y_test, y_pred_proba)

    if toPrint == True:
        # test pvals 1st and 2nd set:
        print '\n'
        print 'num actual positives = %s' % sum(y)
        print 'num actual negatives = %s' % (len(y) - sum(y))
        print '\n'
        print '###### performance #######'
        print 'precision:', precision
        print 'recall:', recall
        print 'F1:', F1
        print 'training accuracy:', train_acc
        print 'test accuracy:', test_acc
        print 'random accuracy would be %s' % rand_acc
        print '##########################'
        print '\n'

    return model, fdf, X, y, X_names, y_name, X_train, X_test, y_train, y_test, train_acc, test_acc, rand_acc, y_pred, y_pred_proba



def build_ML_regression(data, features, labelcol='nyearsParkinsons', toPlot=[0], toPrint=True, MLexcludecols=[], modelType ='linearregression', featureToMean=[], sampleBalance=False, sampleBalanceParams=sampleBalanceDefaultParams):
    '''
    This will run a regression model on the parkinsons dataframe.
    Ugly to have this and the age corrected version. Need to reconcile them.
    For regression models. to do classification, use build_ML_model
    Should combine these later too..

    '''

    # build features dataframe:
    fdf = data[features]

    fdf = convert_features_to_numbers(fdf)

    # drop nas:
    len1 = len(fdf)
    fdf = fdf.dropna()
    len2 = len(fdf)
    print 'dropped %s rows to remove all nas from data' % (len1 - len2)

    # optionally boil each patient down to his mean:
    # (note, will also exclude this column & turn it into the index!)
    if len(featureToMean) > 0:
        fdf = groupby_col_and_avg_other_cols(fdf, featureToMean)
        features.remove(featureToMean[0])

        print '# positive in labelcol: ', fdf[labelcol].sum()
        print '# total in labelcol: ', len(fdf[labelcol])

    # optionally resample after boiling down to mean:
    if sampleBalance == True:
        # define the columns to sample balance on:
        distcol = sampleBalanceParams['distcol'] # 'age'
        splitcol = sampleBalanceParams['splitcol'] # 'hasParkinsons'
        nbins = sampleBalanceParams['nbins'] # 10
        nResamples = sampleBalanceParams['nResamples'] #600
        splitVal_resample = sampleBalanceParams['splitVal_resample'] # False
        splitVal_guide = sampleBalanceParams['splitVal_guide'] # True
        # resample non-Park to same age distribution as Parkinsons:
        df_resampled, df_guide, df_resample = resample_to_match_distribution(fdf, distcol, splitcol, splitVal_resample, splitVal_guide, nbins, nResamples)

        # test pval (we want this to not be significant):
        a = df_resampled[distcol].dropna().values
        b = df_guide[distcol].dropna().values
        p1 = ranksums(a, b)
        print 'pval for resampling (want nonsignificant): ', p1[1]

        # combine back the datasets:
        fdf = df_resampled.append(df_guide)

    # remove cols to exclude from ML (but that were needed for processing)
    if len(MLexcludecols) > 0:
        for col in MLexcludecols:
            fdf = fdf.drop(col, axis=1)
            features.remove(col)

    # prep feature matrix for machine learning:
    features_df, X, y, X_names, y_name, X_train, X_test, y_train, y_test = prep_memory_features_for_machine_learning(fdf, features, labelcol, convert_features_to_nums=False, toStandardScale=False)

    ######### Machine learning #########
    if modelType == 'linearregression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        importances = np.array(model.coef_)
#        print importances

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

#    plt.scatter
#    model.predict(X_test)

    # visualize outputs:
    if toPlot[0] == 1:
#        plot_feature_importances_randforest(model, X_names)
        plot_feature_importances(X_names, importances)


    return model, fdf, X, y, X_names, y_name, X_train, X_test, y_train, y_test, y_pred, y_pred_train




#############################
## Visualization functions ##
#############################



def plot_feature_importances_randforest(model, X_names, useFeatureNames=True):
    '''
    Builds barplot of feature importances for random forest.
    model should be a randomForest model, already trained.
    X_names are the names of the features (np array)
    importances = model.feature_importances_ for random forest
    importances = model.get_params() for logistic regression
    should deprecate this one...
    '''

    # get nice feature names for plot:
    if useFeatureNames:
        fnames = feature_names(X_names)
    else:
        fnames = X_names

    importances = model.feature_importances_
    indices = np.argsort(importances)#[::-1]
    plt.figure(figsize=(3, 7))
    plt.title('Feature importances')
    plt.barh(range(len(fnames)), importances[indices], align='center', )
    plt.ylim([-1, len(fnames)])
    plt.yticks(range(len(fnames)), fnames[indices])
    plt.xticks(rotation=90)
#    for f in range(features):
#        print ("%2d) %-*s %f" % (f+1, 30, features[f], importances[indices[f]]))
    return plt


def plot_feature_importances(X_names, importances, useFeatureNames=True):
    '''
    Need to fix this code..
    Builds barplot of feature importances for random forest.
    model should be a randomForest model, already trained.
    X_names are the names of the features (np array)
    importances = model.feature_importances_ for random forest
    importances = model.get_params() for logistic regression
    '''

    # get nice feature names for plot:
    if useFeatureNames:
        fnames = feature_names(X_names)
    else:
        fnames = X_names

    indices = np.argsort(importances)#[::-1]
    plt.figure(figsize=(3, 7))
    plt.title('Feature importances')
    plt.barh(range(len(fnames)), importances[indices], align='center', )
    plt.ylim([-1, len(fnames)])
    plt.yticks(range(len(fnames)), fnames[indices])
    plt.xticks(rotation=90)
#    for f in range(features):
#        print ("%2d) %-*s %f" % (f+1, 30, features[f], importances[indices[f]]))
    return plt


def display_num_nulls_per_column(df):
    numnulls = df.isnull().sum()
    pd.set_option('display.max_rows', len(numnulls))
    numnulls.sort_values(inplace=True, ascending=True)
    print 'Number of nulls per column:\n'
    print numnulls


def squaregridhistplot(features_df):
    '''
    Plots a grid of plots, each row & col corresponding to a column in the dataframe, with contour maps for each pair & hists on the diagonal
    From: http://stanford.edu/~mwaskom/software/seaborn-dev/tutorial/distributions.html
    '''
    g = sns.PairGrid(features_df)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)
    return g


def render_confusion_matrix(y_true, y_pred, pos_class=True, neg_class=False):
    '''
    Code adapted from Python Machine Learning book
    name_pos_class is name of the positive class (string)
    name_neg_class is name of the negative class (string)
    Usually put y_test as y_true input
    '''

    #ax.set_axis_bgcolor('white')

    # set style:
    sns.set(style="white", color_codes=True, font_scale=1.5)
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # make plot:
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    # fix labels:
    ax.set(xticklabels=['', neg_class, pos_class, ''])
    ax.set(yticklabels=['', neg_class, pos_class, ''])

    plt.tight_layout()
    plt.show()

    # return to default (this is a hack..)
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)


def plot_roc_curve(y_true, y_predictedprobs, startNewPlot=True, withLabel=True):
    '''
    Plots an roc curve.

    For random forest:
    y_predictedprobs = model.predict_proba(X_test)[:,1]
    '''

    # set style:
    sns.set(style="white", color_codes=True, font_scale=1.5)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_predictedprobs)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    if startNewPlot:
        plt.figure()


    # Plot of a ROC curve for a specific class
    if withLabel:
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    else:
        plt.plot(fpr, tpr)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    err = 0.01
    plt.xlim([-err, 1])
    plt.ylim([0.0, 1+err])
    plt.axes().set_aspect('equal')
#    plt.show()

    # return to default (this is a hack..)
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    return fpr, tpr, thresholds


def plot_roc_curves_with_mean(y_trues, y_pred_probas):
    '''
    y_trues and y_pred_probas are lists of length nIters,
    with a y_true and a y_predicted_probability vector in
    each list element (as come out of a machine learning model)
    '''

    nIters = len(y_trues)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    plt.figure()

    for iter in range(nIters):
       probas = y_pred_probas[iter]
       y_true = y_trues[iter]
       fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, probas)
       mean_tpr += np.interp(mean_fpr, fpr, tpr)
       mean_tpr[0] = 0.0
       roc_auc = sklearn.metrics.auc(fpr, tpr)
       plot_roc_curve(y_true, probas, startNewPlot=False, withLabel=False)

    # determine mean line:
    mean_tpr /= nIters
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
            label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], 'k-')
    plt.legend(loc="lower right")
    plt.show()

    # return the style:
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)



#############################
## Miscellaneous functions ##
#############################

def convert_boolean_col_to_int(df, col):
    '''
    This function deals with a (potential bug?) problem in pandas
    When doing groupby, sometimes boolean columns are dropped, as
    they seem to not be treated as numbers (sometimes not - i'm not
    sure why). This function will take a boolean column in df and
    will convert it to integer, and will leave any other values
    alone (i.e., nans will stay as nans).

    '''
    df = df.copy()
    S = df[col].copy()
    S[S==False] = S[S==False].astype(int)
    S[S==True] = S[S==True].astype(int)
    df[col] = S
    return df

def groupby_col_and_avg_other_cols(df, col, keepColinDf=False):
    '''
    dataframe must be all numeric (aside from 'col' column)
    this groups by values in 'col', and then averages the values
    for all other columns corresponding to each unique value in 'col'.
    will keep 'col' in the output df if keepColinDf is true
    '''
    grouped = df.groupby(col)
    df = grouped.apply(lambda x: x.mean())

    # note, the index of the output df is the unique vals from col
    # this option makes them a new column as well (with same name)
    if keepColinDf:
        df[col] = df.index
    return df

def move_col_to_end_of_df(df, colname):
    '''
    moves column colname to be the last column of dataframe df
    '''
    col = df[colname]
    df = df.drop(colname, axis=1)
    df[colname] = col
    return df


def convert_regression_coefs_to_pdSeries(coef_, X_names):
    inlist = coef_.tolist()[0]
    index = X_names.tolist()
    S = pd.Series(inlist, index=index)
    return S


def test():
    return 1


def column_ttests(df, ttestcol, ttestcolCutoff=0.5):
    '''
    Performs ttest of all columns in df (except for ttestcol), against
    values of ttesetcol, where the values are split into positive and
    negative classes based on ttestcol being above or below
    ttestcolCutoff value.
    e.g., ttestcol = 'hasParkinsons',
    df = features_df
    NOTE: will throw an error if any of the columns are not numerical.
    '''

    # split data into low & high categories based on ttestcol vals:
    catlow = df[df[ttestcol] < ttestcolCutoff]
    cathigh = df[df[ttestcol] >= ttestcolCutoff]

    # create list of columns, not including the ttestcol:
    testcols = df.columns.tolist()
    testcols.pop(testcols.index(ttestcol))

    tstats = []
    pvals = []
    for feature in testcols:
#        print 'feature ======= %s' % feature
        t, p = ttest_ind(catlow[feature].dropna(), cathigh[feature].dropna())
        tstats.append(t)
        pvals.append(p)
        #scipy.stats.ranksums()[source]

    ttestresults = pd.DataFrame({'pvals':pvals,'tstats':tstats},index=testcols)
    ttestresults = ttestresults.sort_values('pvals')
    return ttestresults # testcols, pvals, tstats


def resample_to_match_distribution(df, distcol, splitcol, splitVal_resample, splitVal_guide, nbins, nResamples):
    '''
    This will take a dataframe, split it into two parts based on values
    in splitcol (which must only have 2 values in it), and will then
    resample rows from the resulting df_resample dataframe. The resampling
    will be done such that the distribution in column distcol matches
    the distribution of values in distcol in the df_guide dataframe.
    This is intended to help deconfound variables: e.g., if age is a
    confound for hasParkinsons, run it like this:

    df = features_df
    distcol = 'age'
    splitcol = 'hasParkinsons'
    splitVal_resample = False
    splitVal_guide = True
    nbins = 10
    nResamples = 100

    df = the dataframe to work on
    distcol = the column in df that should have matching distributions
    splitcol = the column in df that will be split into resample and guide
    splitVal_resample = value in splitcol defining the df to be resampled
    splitVal_guide = value in splitcol for df whose dist should be matched
    nResamples = # of rows from the resample df to be output

    resamples done without replacement. nans are not included in distribution.

    outputs:
    df_resampled = the resampled version of df_resample
    df_guide = the df with splitVal_guide vals in splitcol
    df_resample = the df to resample
    '''

    ### split dataframe into df_resample and df_guide:
    df_resample = df[df[splitcol] == splitVal_resample]
    df_guide = df[df[splitcol] == splitVal_guide]

    ### take a histogram of df_guide, to get density distribution:
    guidevals = df_guide[distcol].values
    guidevals = guidevals[~np.isnan(guidevals)]
    hist, binedges = np.histogram(guidevals, bins=nbins)

    ### create weights vector:

    # (1-row per row in df_resample, with a weight on that row
    # determined by the density distribution of df_guide)
    resamplevals = df_resample[distcol].values
    wts = np.zeros(resamplevals.shape)

    # find the vals within each histogram bin:
    for n, histval in enumerate(hist):
        leftedge = binedges[n]
        rightedge = binedges[n+1]
        goodinds = np.where((resamplevals > leftedge) & \
                               (resamplevals <= rightedge))
        wts[goodinds] = histval

    ### normalize weights:
    wts = wts/sum(wts)

    ### sample the indices of resamplevals with the given weights:
    #samples = np.array(['a','b','c','d'])
    allinds = np.arange(len(resamplevals))
    indsamples = np.random.choice(allinds, size=nResamples, replace=False, p=wts)

    ### output df_resample with only the sampled rows:
    # (this makes the index point to allinds, so they must be identical!)

    #df_resample = df_resample.reindex(allinds)
    #assert (numpy.array_equal(df_resample.index.values, allinds)), 'there is a problem with\the indices. sampling won''t come out right.'

    df_resampled = df_resample.iloc[indsamples,:]

    return df_resampled, df_guide, df_resample


@contextmanager
def suppress_stdout():
    '''
    From http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    Suppresses print statements for a function call.

    Use like:

    print "You can see this"
    with suppress_stdout():
        print "You cannot see this"
    print "And you can see this again"

    '''

    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def check_for_string_in_dfcol(df, col):
    '''
    Check if a column of a dataframe contains any strings
    '''
    hasString = False
    for val in df[col].values:
        if type(val)==str:
            hasString = True

    return hasString







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


# convert nas to large negative #:
#naval = -999999
#data.fillna(value=naval)








