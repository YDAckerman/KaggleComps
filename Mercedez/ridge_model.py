import pandas as pd
import numpy as np
from numpy import dot
import help_funs as hf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


## load the training set
train = pd.read_csv("train.csv", index_col = "ID")
test = pd.read_csv("test.csv", index_col = "ID")

## load in sample submissions so we can get the
## correct indices
sample_preds = pd.read_csv("sample_submission.csv")

## ugh - all the columns need to match for the appending
y = pd.DataFrame(pd.Series([0 for i in range(test.shape[0])], index = test.index),
                 columns = ['y'])
test.insert(0, 'y', y)

## append
full = train.append(test)

## create dummy variables out of the categorical ones
for i in [0,1,2,3,4,5,6,8]:
    colname = "X" + str(i)
    col = full[colname]
    full = full.drop(colname, 1)
    full = pd.concat([full, hf.dummify(col, colname)], axis = 1)

train = full.iloc[range(train.shape[0]), :]
test = full.drop(range(train.shape[0]), axis = 0)

# singular matrix - ridge for now
res = []
folds = hf.k_fold(train.shape[0], 5)
for l in range(800, 1500, 20):
    error_k = []
    for index in range(len(folds)):
        fold = folds[index]
        ## create the train and test sets from this fold
        train_k, test_k = hf.make_fold(fold, train)
        ## create response/feature DataFrames
        Y_k, X_k = hf.to_XY(train_k, 0, range(1,train_k.shape[1]))
        y_k, x_k = hf.to_XY(test_k, 0, range(1,test_k.shape[1]))
        ## standardize
        X_k, Y_k, x_k, y_k = hf.standardize_mats(X_k,Y_k,x_k,y_k)
        ## calculate the Beta coefficient with this lambda (l) value
        try:
            B_k = hf.ridge_coef(X_k, Y_k, l)
            ## calculate the training error
            e_hat_k = y_k - dot(x_k, B_k)
            error_k.append(np.sum(e_hat_k**2))
        except:
            print "l: " + str(l)
            print "fold: " + str(index)
    res.append((l, np.sum(error_k)))

res = pd.DataFrame.from_records(res, columns = ["l", "err"])
cvDat = pd.DataFrame(res["err"], index = res["l"])
cvDat.plot()

## use best B_k to calculate submission predictions
best_l = 1400
Y, X = hf.to_XY(train, 0, range(1, train.shape[1]))
y, x = hf.to_XY(test, 0, range(1, test.shape[1]))
Y_mean = Y.mean(axis = 0)
X, Y, x, y = hf.standardize_mats(X,Y,x,y)
B_hat = hf.ridge_coef(X, Y, best_l)
Y_hat = list(dot(X, B_hat) + Y_mean)
y_hat = list(dot(x, B_hat) + Y_mean)
preds = pd.Series(Y_hat + y_hat, index = full.index + 1)
preds = preds[sample_preds["ID"]]

preds.to_csv("prediction1.csv")
