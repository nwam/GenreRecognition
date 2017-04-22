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
TRAIN_PERCENT = 0.9
N_PER_YEAR = 1000 # number of instances per class
YEAR_START = 1965
YEAR_END = 2010
YEARS = list(range(YEAR_START,YEAR_END+1))

y = [] # output
X = [] # input 

y_test = []
y_train = []
X_test = []
X_train = []

# build a genre counter to have the same number of inputs per genre
year_count = {}
for year in YEARS:
    year_count[year] = 0

# get x's and y's from data file
# file contains one instance on each line with the format
#   y,x1,x2,...,xn
print("[gathering data]")
with open(DATA_FILE, "r") as f:
    for line in f:
        values = line.split(DELIM)


        year = int(values[0])
        if year not in YEARS:
            continue

        if year_count[year] < N_PER_YEAR:
            year_count[year] = year_count[year] + 1
            
            X_entry = []
            for value in values[2:]:
                X_entry.append(float(value))

            if year_count[year] < N_PER_YEAR*TRAIN_PERCENT:
                y_train.append(year)
                X_train.append(X_entry)
                     
            else: # add to test
                y_test.append(year)
                X_test.append(X_entry)

print(year_count)

# preprocessing
print("[preprocessing data]")
X_train = sklearn.preprocessing.normalize(X_train)
X_test = sklearn.preprocessing.normalize(X_test)


# split data into train and test
#print("[splitting data]")
#split_index = int(len(y) * TRAIN_PERCENT)
#y_train = y[:split_index]
#X_train = X[:split_index]
#y_test  = y[split_index:]
#X_test  = X[split_index:]
#print(len(y_train), len(y_test))

# train the classifier (svm)
print("[training classifier]")
classifier = SVR(kernel='rbf', C=20000)
classifier.fit(X_train, y_train)

# test the classifier
print("[testing classifier]")
y_predict = classifier.predict(X_test)
y_predict = [round(y) for y in y_predict]

print("Accuracy: %.2f%%" % (100*classifier.score(X_test, y_test)))
print("MSE: %.2f years" % (mean_squared_error(y_test, y_predict)))

conf_mtx = confusion_matrix(y_test, y_predict, labels=YEARS)
np.savetxt(OUTPUT_FILE, conf_mtx, delimiter=",")

# make graphs
print("[creating cool visualizations]")
plt.figure()
plots.plot_confusion_matrix(conf_mtx, classes=YEARS ,normalize=True, savefile=PLOT_FILENAME, print_values=False)

# Time
print("Execution time: %.2f seconds" % (time.time()-start_time))
