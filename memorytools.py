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
from numpy import nan

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
from sklearn.ensemble import RandomForestClassifier

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

    print 'all memory features: %s' % all_memory_features
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


#################################
## Machine Learning, data prep ##
#################################

def convert_features_to_numbers(features_df):
    '''
    Prep step of particular features, which are categorical (but ordered) and should be converted to ordinal or cardinal #'s - this converts them to numbers for import to machine learning model.
    Should be fixed to deal better with nans.
    '''
    df = pd.DataFrame.copy(features_df)

    def ordinate_categorical_col(df, column, code):
        '''
        convert a categorical (but ordered) column to #'s, based on a manually determined conversion code.
        '''
        def assign_code(code, colval):
#            print 'COLVAL IS!!!!!!!!! %s' % colval
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

    ### Education by # years post-middleschool:
    education_code = {'Some high school':2,
    'High School Diploma/GED':4, '2-year college degree':6,
    'Some college':6, '4-year college degree':8,
    'Some graduate school':10, "Master's Degree":10,
    'Doctoral Degree':13}
    ### Smartphone by difficulty description:
    smartphone_code = {'Very easy':1, 'Easy':2,
    'Neither easy nor difficult':3, 'Difficult':4,
    'Very Difficult':5}
    ### Gender (binary ordinates):
    gender_code = {'Male':1, 'Female':0}
    ### Phone Usage (what does this mean?):
    phoneUsage_code = {'false':0, 'Not sure':1, 'true':2}
    ### Phone Info (the phone used) (encoded as screen size):
    phoneInfo_code = {'iPhone 5s (GSM)':4.0, 'iPhone 6':4.7,
    'iPhone 6 Plus':5.5}

    ### do feature ordinations:
    fcodes = {'smartphone':smartphone_code,
    'education':education_code,
    'gender':gender_code,
    'phoneUsage':phoneUsage_code,
    'phoneInfo':phoneInfo_code}

    featureschanged = []
    for feature in fcodes:
        if feature in df:
            df = ordinate_categorical_col(df, feature, fcodes[feature])
            featureschanged.append(feature)

#    df = ordinate_categorical_col(df, 'smartphone', smartphone_code)

#    df = ordinate_categorical_col(df, 'education', education_code)
#    df = ordinate_categorical_col(df, 'gender', gender_code)
#    df = ordinate_categorical_col(df, 'phoneUsage', phoneUsage_code)
#    df = ordinate_categorical_col(df, 'phoneInfo', phoneInfo_code)
    print 'Features converted to numbers:\n'
    print featureschanged #smartphone, education, gender, phoneUsage, phoneInfo'

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


def prep_memory_features_for_machine_learning(data, features, labelcol):
    '''
        Uses other ML prep functions to get memory features in and ready for machine learning.
        This takes in the memory data dataframe, the features of interest, and which feature should be the label column (i.e., what's being predicted), and formats all of this correctly for inputting into sklearn.
    '''

    ##################### Preprocess data for machine learning:
    # define features (include the label feature here:
#    features = ['game_score', 'age', 'hasParkinsons']
    features_df = data[features]
    features_df = convert_features_to_numbers(features_df)
    features_df = move_col_to_end_of_df(features_df, 'hasParkinsons')

    # do more processing here, in case of features with lots of nas?

    # drop na rows:
    features_df = features_df.dropna()

    # convert to matrices for machine learning:
    #labelcol = 'hasParkinsons'
    X, y, X_names, y_name = convert_features_df_to_X_and_y_for_machinelearning(features_df, labelcol)

    ##################### Set features up for machine learning:

    # split for cross validation:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    # scale features:
    stdsc = StandardScaler()
    stdsc.fit(X_train)
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    return features_df, X, y, X_names, y_name, stdsc, X_train_std, X_test_std, X_combined_std, y_combined





#############################
## Miscellaneous functions ##
#############################

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

def display_num_nulls_per_column(df):
    numnulls = df.isnull().sum()
    pd.set_option('display.max_rows', len(numnulls))
    numnulls.sort_values(inplace=True, ascending=True)
    print 'Number of nulls per column:\n'
    print numnulls






#######################################
## from Python Machine Learning book ##
#######################################
def plot_decision_regions(X, y, classifier,
                       test_idx=None, resolution=0.02):
    '''
    Plot a 2-column X vs y, given a trained classifier
    '''
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                alpha=1.0, linewidth=1, marker='o',
                s=55, label='test set')



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








