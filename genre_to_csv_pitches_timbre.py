from helper import hdf5_getters

import os
import sys
import glob
import time
import csv
import numpy as np

BASE_DIR = "/home/nwam/4437/project/my_code/data/MillionSongSubset/data/"
OUTPUT = "data_genre_pitches_timbre.csv"
GENRES = ['hip hop', 'rock', 'electronic', 'pop', 'jazz', 'country']

def get_all_files(basedir,ext='.h5') :
    """
    From a root directory, go through all subdirectories
    and find all files with the given extension.
    Return all absolute paths in a list.
    """
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files :
            allfiles.append( os.path.abspath(f) )
    return allfiles

def get_genres(h5, unique=False):
    artist_terms = np.asarray(hdf5_getters.get_artist_terms(h5))
    song_genres = []

    # find genres
    for genre in artist_terms:
        genre = str(genre)[2:-1] 
        if genre in GENRES:
            song_genres.append(genre)

            if(unique):
                if len(song_genres) > 1:
                    return None

    if unique:
        if len(song_genres)==0:
            return None
        else:
            return song_genres[0]
    return song_genres

def pitches_averages(segments_pitches):
    return segments_pitches.mean(axis=0)

def pitches_covariances(segments_pitches):
    # column vectors of segments_pitches
    pitches_cols = []     
    for col in range(len(segments_pitches[0])):
        pitches_cols.append(segments_pitches[:,col])

    # get covariances
    pitches_covs = []
    for i in range(len(pitches_cols)):
        for j in range(i,len(pitches_cols)):
            pitches_cov = np.cov(pitches_cols[i], pitches_cols[j])
            pitches_covs.append(pitches_cov[0][1])

    return pitches_covs

# init file list
allh5 = get_all_files(BASE_DIR, '.h5')

# open output file
open(OUTPUT, 'w').close()
output = open(OUTPUT, 'a')

# init genre counter
genre_count = {}
genre_count[None] = 0
for genre in GENRES:
    genre_count[genre] = 0

for h5filename in allh5:
    # open datafile
    with hdf5_getters.open_h5_file_read(h5filename) as h5:
        
        # get/process data 
        song_genre = get_genres(h5, unique=True)
        if song_genre == None:
            continue;
        song_pitches = np.array(hdf5_getters.get_segments_pitches(h5))
        song_timbre = np.array(hdf5_getters.get_segments_timbre(h5))
        song_pitches_avgs = pitches_averages(song_pitches)
        song_pitches_covs = pitches_covariances(song_pitches)
        song_timbre_avgs = pitches_averages(song_timbre)
        song_timbre_covs = pitches_covariances(song_timbre)

        # count genres
        genre_count[song_genre] = genre_count[song_genre] + 1

        # aggregate data into one row
        row = np.append(song_pitches_avgs, song_pitches_covs).tolist()
        row2 = np.append(song_timbre_avgs, song_timbre_covs).tolist()
        row = [song_genre] + row + row2

        # write data to output
        csv.writer(output).writerow(row)

        # close datafile
        h5.close()

output.close()


print(genre_count)
