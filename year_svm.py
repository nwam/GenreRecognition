import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from matplotlib import style

start_time = time.time()
np.set_printoptions(threshold=np.nan)

DATA_FILE = "data/YearPredictionMSD.txt"
OUTPUT_FILE = "year_confusion.csv"
DELIM = ","
SUBSET_SCALE_DOWN_FACTOR = 1 # =1 to use full dataset
TRAIN_PERCENT = 0.9
DATE_START = 1922
DATE_END = 2012

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
    

# split data into train and test
print("[splitting data]")
split_index = int(len(y) * TRAIN_PERCENT)
y_train = y[:split_index]
X_train = X[:split_index]
y_true  = y[split_index:]
X_test  = X[split_index:]


# build the classifier (svm)
print("[building classifier]")
classifier = linear_model.SGDClassifier(n_iter=100, alpha=0.01) # hinge for unbalanced data
classifier.fit(X_train, y_train)


# test the classifier
print("[testing classifier]")
print("Mean accuracy: %.2f%%" % (100*classifier.score(X_test, y_true)))

y_predict = classifier.predict(X_test)
conf_mtx = confusion_matrix(y_true, y_predict, labels=np.arange(DATE_START, DATE_END, 1))
np.savetxt(OUTPUT_FILE, conf_mtx, delimiter=",")
    
    

# Time
print("Execution time: %.2f seconds" % (time.time()-start_time))
