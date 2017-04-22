# T
# This script works with the data generated/processed from get_all_the_data.py
from helper import plots

import numpy as np
import matplotlib.pyplot as plt
import time
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, mean_squared_error

import matplotlib.pyplot as plt

start_time = time.time()
np.set_printoptions(threshold=np.nan)

DATA_FILE = "data/literall_all.csv"
OUTPUT_FILE = "genre_confusion.csv"
PLOT_FILENAME = "genre_confusion.png"
DELIM = ","
TRAIN_PERCENT = 0.9
N_PER_GENRE = 1000 # number of instances per class
#GENRES = ['hip hop', 'rock', 'electronic', 'pop', 'jazz', 'folk']
GENRES = ['hip hop', 'rock', 'electronic', 'pop', 'jazz', 'country', 'metal', 'reggae', 'r&b', 'folk']

y = [] # output
X = [] # input 

y_test = []
y_train = []
X_test = []
X_train = []

# build a genre counter to have the same number of inputs per genre
genre_count = {}
for genre in GENRES:
    genre_count[genre] = 0

# get x's and y's from data file
# file contains one instance on each line with the format
#   y,x1,x2,...,xn
print("[gathering data]")
with open(DATA_FILE, "r") as f:
    for line in f:
        values = line.split(DELIM)


        genre = values[0]
        year = int(values[1])
        if genre not in GENRES:
            continue

        if genre_count[genre] < N_PER_GENRE:
            genre_count[genre] = genre_count[genre] + 1
            
            X_entry = []
            for value in values[2:]:
                X_entry.append(float(value))

            if genre_count[genre] < N_PER_GENRE*TRAIN_PERCENT:
                y_train.append(genre)
                X_train.append(X_entry)
                     
            else: # add to test
                y_test.append(genre)
                X_test.append(X_entry)

print(genre_count)

# preprocessing
print("[preprocessing data]")
X_train = sklearn.preprocessing.normalize(X_train)
X_test = sklearn.preprocessing.normalize(X_test)

# train the classifier (svm)
print("[training classifier]")
classifier = SVC(kernel='rbf', C=5000)
classifier.fit(X_train, y_train)

# test the classifier
print("[testing classifier]")
y_predict = classifier.predict(X_test)

print("Accuracy: %.2f%%" % (100*classifier.score(X_test, y_test)))

conf_mtx = confusion_matrix(y_test, y_predict, labels=GENRES)
np.savetxt(OUTPUT_FILE, conf_mtx, delimiter=",")

# make graphs
print("[creating cool visualizations]")
plt.figure()
plots.plot_confusion_matrix(conf_mtx, classes=GENRES ,normalize=True, savefile=PLOT_FILENAME)

# Time
print("Execution time: %.2f seconds" % (time.time()-start_time))
