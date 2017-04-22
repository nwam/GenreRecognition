import numpy as np
from matplotlib import style

DATA_FILE = "data/YearPredictionMSD.txt"
OUTPUT_FILE = "data_by_year.csv"
DELIM = ","

Y_LOW = 1922
Y_HIGH = 2011
y = np.zeros(Y_HIGH-Y_LOW+1)

with open(DATA_FILE, "r") as f:
    for line in f:
        values = line.split(DELIM)
        year = int(values[0])
        y[year-Y_LOW] = y[year-Y_LOW] + 1

np.savetxt(OUTPUT_FILE, y, delimiter=",")
