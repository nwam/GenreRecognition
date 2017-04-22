from helper import plots

import numpy as np
import matplotlib.pyplot as plt
import time
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, mean_squared_error

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import optunity
import optunity.metrics

start_time = time.time()
np.set_printoptions(threshold=np.nan)

DATA_FILE = "data/data_genre_pitches.csv"
OUTPUT_FILE = "genre_confusion.csv"
PLOT_FILENAME = "genre_confusion.png"
DELIM = ","
SUBSET_SCALE_DOWN_FACTOR = 1 # =1 to use full dataset, >1 to not
TRAIN_PERCENT = 0.9
GENRES = ['hip hop', 'rock', 'electronic', 'pop', 'jazz', 'country']

y = [] # output
X = [] # input 

# get x's and y's from data file
# file contains one instance on each line with the format
#   y,x1,x2,...,xn
print("[gathering data]")
with open(DATA_FILE, "r") as f:
    i = -1
    for line in f:
        i = i+1 
        if i%SUBSET_SCALE_DOWN_FACTOR!=0:
            continue

        values = line.split(DELIM)
        genre = values[0]
        if genre not in GENRES:
            continue

        y.append(genre)

        X_entry = []
        for value in values[1:]:
            X_entry.append(float(value))
        X.append(X_entry)

# preprocessing
print("[preprocessing data]")
X = sklearn.preprocessing.normalize(X)

# split data into train and test
print("[splitting data]")
split_index = int(len(y) * TRAIN_PERCENT)
y_train = y[:split_index]
X_train = X[:split_index]
y_test  = y[split_index:]
X_test  = X[split_index:]
print(len(y_train), len(y_test))

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
