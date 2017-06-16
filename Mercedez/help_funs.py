import pandas as pd
import numpy as np
from numpy import var, mean, sqrt, dot, identity
from numpy.linalg import inv
from numpy.random import permutation


def dummify(column, name):
    """
    Creates dummy variables from a categorial variable
    """
    colvals = column.unique()
    tmpDict = dict([(j,i) for i,j in enumerate(colvals)])
    dummies = []
    for j in column:
        row = [0 for k in range(len(colvals))]
        row[tmpDict[j]] = 1
        dummies.append(tuple(row))
    labels = [name + str(j) for j in colvals]
    return pd.DataFrame.from_records(dummies, columns = labels)


def scale_mat(mat):
    """
    scales and centers a numpy matrix
    """
    return (mat - mat.mean(axis = 0)) / sqrt(mat.var(axis = 0))

def k_fold(nrows, ks, seed = 0):
    """
    return a list of folds
    """
    np.random.seed(seed = seed)
    indices = permutation(nrows)
    folds = []
    for k in range(ks):
        folds.append([ j for i,j in enumerate(indices) if i % ks == k])
    return folds
    
def make_fold(fold, train):
    """
    creates a train and test set given a fold
    """
    indices = np.array(fold)
    train_k = train.drop(indices, 0)
    test_k = train.iloc[indices, :]
    return train_k, test_k

def to_XY(data, y_col, x_cols):
    """
    Returns response and feature dataframes
    """
    Y = data.iloc[:, y_col]
    X = data.iloc[:, x_cols]
    return Y, X

def ridge_coef(X, Y, l):
    """
    calculate the ridge coefficients
    """
    B = dot(dot(inv(dot(X.transpose(), X) + l * identity(X.shape[1])),
                      X.transpose()), Y)
    return B

def standardize_mats(X, Y, x, y, scale_y = False, center_y = True,
                     scale_x = True):
    if center_y:
        y -= Y.mean(axis = 0)
        Y -= Y.mean(axis = 0)
    if scale_y:
        y /= sqrt(Y.var(axis = 0))
        Y /= sqrt(Y.var(axis = 0))
    if scale_x:
        # first remove any X columns with variance 0
        zeroVar = X.var(axis = 0) != 0
        X = X.loc[:, list(zeroVar)]
        x = x.loc[:, list(zeroVar)]
        x = (x - X.mean(axis = 0)) / sqrt(X.var(axis = 0))
        X = (X - X.mean(axis = 0)) / sqrt(X.var(axis = 0))
    return X,Y,x,y
