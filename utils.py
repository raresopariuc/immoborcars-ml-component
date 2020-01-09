import pandas as pd
import numpy as np
from sklearn import preprocessing

def gettingHousesData(path, colsX, colsLabely):
    df = pd.read_csv(path)
    cols = getHousesCols()
    for col in cols:
        df[col] = pd.factorize(df[col])[0]
    X = df[colsX]
    y = df[colsLabely]
    return df, X, y

def gettingCarsData(path, colsX, colsLabely):
    df = pd.read_csv(path)
    cols = getCarsCols()
    for col in cols:
        df[col] = pd.factorize(df[col])[0]
    X = df[colsX]
    y = df[colsLabely]
    return df, X, y

def getHousesCols():
    cols = ['YearBuilt', 'TotalBasementArea', 'NumberOfBathrooms', 'TotalNumberOfRooms']
    
    return cols

def getCarsCols():
    cols = ['Mileage', 'YearOfManufacture', 'Power']
    
    return cols