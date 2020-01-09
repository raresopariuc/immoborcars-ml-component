from flask import Flask
from flask import jsonify, request, Response
from flask_cors import CORS, cross_origin

import json
import pickle

import pandas
import numpy as np

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn import preprocessing

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

import utils

globalHousesPath = 'housesModel_lowerPrices.csv'
globalCarsPath = 'carsModel.csv'

app = Flask(__name__)
CORS(app)

@app.route("/train-gradient-boosting-regressor-for-houses")
def trainGradientBoostingRegressorForHouses():
    _, X, y = utils.gettingHousesData(path=globalHousesPath, colsX=utils.getHousesCols(), colsLabely=['Label'])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1)

    gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.001, max_features="sqrt", max_depth=10, loss='lad', verbose=1)
    gbr.fit(X_train, y_train)

    filename = 'gbr_houses_model.sav'
    pickle.dump(gbr, open(filename, 'wb'))
    return jsonify(result="model trained")

@app.route("/train-gradient-boosting-regressor-for-cars")
def trainGradientBoostingRegressorForCars():
    _, X, y = utils.gettingCarsData(path=globalCarsPath, colsX=utils.getCarsCols(), colsLabely=['Label'])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1)

    gbr = GradientBoostingRegressor(n_estimators=600, learning_rate=0.001, max_features="sqrt", max_depth=10, loss='lad', verbose=1)
    gbr.fit(X_train, y_train)

    filename = 'gbr_cars_model.sav'
    pickle.dump(gbr, open(filename, 'wb'))
    return jsonify(result="model trained")

@app.route("/use-gradient-boosting-regressor-for-houses", methods=['POST'])
def useGradientBoostingRegressorForHouses():
    to_predict = []
    for col in utils.getHousesCols():
        to_predict += [request.json.get(col)]
    to_predict = np.array(to_predict).reshape(1, -1)

    filename = 'gbr_houses_model.sav'
    gbr = pickle.load(open(filename, 'rb'))
    prediction = gbr.predict(to_predict)[0]

    return jsonify(result=str(prediction))

@app.route("/use-gradient-boosting-regressor-for-cars", methods=['POST'])
def useGradientBoostingRegressorForCars():
    to_predict = []
    for col in utils.getCarsCols():
        to_predict += [request.json.get(col)]
    to_predict = np.array(to_predict).reshape(1, -1)

    filename = 'gbr_cars_model.sav'
    gbr = pickle.load(open(filename, 'rb'))
    prediction = gbr.predict(to_predict)[0]

    return jsonify(result=str(prediction))

@app.route("/")
def home():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True)