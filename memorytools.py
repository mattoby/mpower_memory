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


# log in
def login_synapse(username, password):
    syn = synapseclient.Synapse()
    syn.login(username, password) # need to change this, security.
    return syn


# get the memory table up and running:
def load_memory_table_from_synapse(syn):
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

    return syn, memory, memorysyn, filePaths, demographics, demosyn, data




















