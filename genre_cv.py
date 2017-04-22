from helper import plots

import numpy as np
import matplotlib.pyplot as plt
import time
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import optunity
import optunity.metrics

start_time = time.time()
np.set_printoptions(threshold=np.nan)

KERNELS = ['rbf',"poly",'linear']
C_VALS = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
GAMMAS = [1, 0.1, 0.001, 0.0001, 0.00001] 

DATA_FILE = "data/genre.csv"
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

        y.append(values[0])

        X_entry = []
        for value in values[1:]:
            X_entry.append(float(value))
        X.append(X_entry)

# preprocessing
print("[preprocessing data]")
X = sklearn.preprocessing.normalize(X)

# split data into train and test
#print("[splitting data]")
#split_index = int(len(y) * TRAIN_PERCENT)
#y_train = y[:split_index]
#X_train = X[:split_index]
#y_test  = y[split_index:]
#X_test  = X[split_index:]
#print(len(y_train), len(y_test))

# train the classifier and test with kfold
# for differnt C, gamma, and kernels
print("[training/testing classifier]")

best_score  = 0
best_c      = None
best_gamma  = None
best_kernel = None

for kernel in KERNELS:
    for c_val in C_VALS:
        for gamma in GAMMAS:
            train_start_time = time.time() 
            classifier = SVC(kernel=kernel, C=c_val, gamma=gamma)
            scores = cross_val_score(classifier, X, y, cv=5)
            avg_score = scores.mean()
            std_score = scores.std()
                
            print("Accuracy: %0.2f (+/- %0.2f) for %s c=%f gamma=%f in %0.2f s" % (avg_score, std_score*2, kernel,
            c_val, gamma,
            time.time()-train_start_time))

            if avg_score>best_score:
                best_score = avg_score
                best_c = c_val
                best_gamma = gamma
                best_kernel = kernel


print("And the winner is...\n kernel=%s c=%f, gamma=%f" % ( best_kernel, best_c, gamma))

# build the best classifier
#classifier = SVC(kernel=best_kernel, C=best_c, gamma=best_gamma)
#
#
## create confusion matrix with best classifier
#y_predict = classifier.predict(X_test)
#conf_mtx = confusion_matrix(y_test, y_predict, labels=GENRES)
#np.savetxt(OUTPUT_FILE, conf_mtx, delimiter=",")
#
## make graphs
#print("[creating cool visualizations]")
#plt.figure()
#plots.plot_confusion_matrix(conf_mtx, classes=GENRES ,normalize=True, savefile=PLOT_FILENAME)

# Time
print("Execution time: %.2f seconds" % (time.time()-start_time))
