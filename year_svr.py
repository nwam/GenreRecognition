from helper import plots

import numpy as np
import matplotlib.pyplot as plt
import time
import sklearn
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, mean_squared_error

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import optunity
import optunity.metrics

start_time = time.time()
np.set_printoptions(threshold=np.nan)

DATA_FILE = "data/YearPredictionMSD.txt"
OUTPUT_FILE = "year_confusion.csv"
PLOT_FILENAME = "year_confusion.png"
DELIM = ","
SUBSET_SCALE_DOWN_FACTOR = 1000 # =1 to use full dataset, >1 to not
TRAIN_PERCENT = 0.9
DATE_START = 1922
DATE_END = 2012
YEAR_LABELS = np.arange(DATE_START, DATE_END, 1)

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

        y.append(int(values[0]))

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
classifier = SVR(kernel='rbf', C=100000, epsilon=0.000001)
classifier.fit(X_train, y_train)

# test the classifier
print("[testing classifier]")
y_predict = classifier.predict(X_test)
y_predict = [int(round(x)) for x in y_predict]

print("Accuracy: %.2f%%" % (100*classifier.score(X_test, y_test)))
print("MSE: %.2f years" % (mean_squared_error(y_test, y_predict)))

conf_mtx = confusion_matrix(y_test, y_predict, labels=YEAR_LABELS)
np.savetxt(OUTPUT_FILE, conf_mtx, delimiter=",")

# make graphs
print("[creating cool visualizations]")
plt.figure()
plots.plot_confusion_matrix(conf_mtx, classes=YEAR_LABELS, normalize=True)
plt.savefig(PLOT_FILENAME, bbox_inches='tight')

# Time
print("Execution time: %.2f seconds" % (time.time()-start_time))
