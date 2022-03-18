import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib
import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import shap
import requests

class IrisSpecies(BaseModel):
    AGE: float
    DAYS_EMPLOYED: float
    DAYS_ID_PUBLISH: float
    REGION_RATING_CLIENT: float
    REG_CITY_NOT_WORK_CITY: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DAYS_LAST_PHONE_CHANGE: float
    CODE_GENDER_F: float
    Working: float
    Higher_Education: float
    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    AMT_ANNUITY: float

class IrisModel:
    def __init__(self):
        self.df = pd.read_csv('creditdf.csv')
        self.model_fname_ = 'f_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            self.model = self._train_model()
            joblib.dump(self.model, self.model_fname_)

    def _train_model(self):
        self.df= self.df.drop(['Unnamed: 0'], axis=1)
        self.df = self.df.drop(['Client'], axis=1)
        X = self.df.drop('TARGET', axis=1)
        y = self.df['TARGET']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        rf = RandomForestClassifier(max_depth=4, random_state=0, class_weight='balanced')
        model = rf.fit(X_train, y_train)
        return model

    def predict_species(self, AGE, DAYS_EMPLOYED,DAYS_ID_PUBLISH,REGION_RATING_CLIENT,REG_CITY_NOT_WORK_CITY,EXT_SOURCE_2,EXT_SOURCE_3,DAYS_LAST_PHONE_CHANGE,CODE_GENDER_F,Working,Higher_Education,AMT_CREDIT,AMT_INCOME_TOTAL,AMT_ANNUITY):
        data_in = [[AGE, DAYS_EMPLOYED,DAYS_ID_PUBLISH,REGION_RATING_CLIENT,REG_CITY_NOT_WORK_CITY,EXT_SOURCE_2,EXT_SOURCE_3,DAYS_LAST_PHONE_CHANGE,CODE_GENDER_F,Working,Higher_Education,AMT_CREDIT,AMT_INCOME_TOTAL,AMT_ANNUITY]]
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in).max()
        return prediction[0], probability
