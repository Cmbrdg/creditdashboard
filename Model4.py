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

st.title('Credit dashboard')
# st.write("""# Explore different variables""")
st.write("""Explore different variables""")

train_df = pd.read_csv('/Users/riina//Downloads/creditdf.csv')

clients = train_df.head(100)
clients = clients[['Client']]

selected_status = st.selectbox(
    'Select variable',
    options=['AGE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL','DAYS_EMPLOYED'])

client_number = st.sidebar.selectbox(
    'Select Client',
    (clients))

st.sidebar.write('You selected: ', client_number)

rslt_df = train_df[(train_df['Client'] == client_number)]
gender=rslt_df[['CODE_GENDER_F']] == 0
gender=np.array(gender)
education=rslt_df[['Higher_Education']]==0
education=np.array(education)
working=rslt_df[['Working']]==0
working=np.array(working)

if gender == True:
    st.sidebar.write("Gender: Male")
if gender == False:
    st.sidebar.write("Gender: Female")

if education == True:
    st.sidebar.write("Higher education: No")
if education == False:
    st.sidebar.write("Higher education: Yes")

if working == True:
    st.sidebar.write("Income type: Other")
if working == False:
    st.sidebar.write("Income type: Working")

bday1=np.mean(train_df[train_df["TARGET"] == 1])
bday1=bday1['AGE']
bday2=np.mean(train_df[train_df["TARGET"] == 0])
bday2=bday2['AGE']
cbday=float(rslt_df['AGE'])

if selected_status == 'AGE':
    st.write('Candidate: ', cbday)
    st.write('Non-Falter mean: ', bday2)
    st.write('Falter mean: ', bday1)

credit1=np.mean(train_df[train_df["TARGET"] == 1])
credit1=credit1['AMT_CREDIT']
credit2=np.mean(train_df[train_df["TARGET"] == 0])
credit2=credit2['AMT_CREDIT']
ccredit=float(rslt_df['AMT_CREDIT'])

if selected_status == 'AMT_CREDIT':
    st.write('Candidate: ', ccredit)
    st.write('Non-Falter mean: ', credit2)
    st.write('Falter mean: ', credit1)

income1=np.mean(train_df[train_df["TARGET"] == 1])
income1=income1['AMT_INCOME_TOTAL']
income2=np.mean(train_df[train_df["TARGET"] == 0])
income2=income2['AMT_INCOME_TOTAL']
cincome=float(rslt_df['AMT_INCOME_TOTAL'])

if selected_status == 'AMT_INCOME_TOTAL':
    st.write('Candidate: ', cincome)
    st.write('Non-Falter mean: ', income2)
    st.write('Falter mean: ', income1)

employed1=np.mean(train_df[train_df["TARGET"] == 1])
employed1=employed1['DAYS_EMPLOYED']
employed2=np.mean(train_df[train_df["TARGET"] == 0])
employed2=employed2['DAYS_EMPLOYED']
cemployed=float(rslt_df['DAYS_EMPLOYED'])

if selected_status == 'DAYS_EMPLOYED':
    st.write('Candidate: ', cemployed)
    st.write('Non-Falter mean: ', employed2)
    st.write('Falter mean: ', employed1)

if selected_status == 'AGE':
    hist_data = [train_df[train_df["TARGET"] == 1]["AGE"], train_df[train_df["TARGET"] == 0]["AGE"]]
    group_labels = ["Falter", "Non-Falter"]
    fig = ff.create_distplot(hist_data, group_labels,show_hist=False)
    fig['layout'].update(title={"text": 'Distribution of clients'}, xaxis_title="Days of Birth", yaxis_title="density")
    fig.add_vline(x=cbday, line_width=1, line_color="black")
    st.plotly_chart(fig, use_container_width=True)

if selected_status == 'AMT_CREDIT':
    hist_data = [train_df[train_df["TARGET"] == 1]["AMT_CREDIT"], train_df[train_df["TARGET"] == 0]["AMT_CREDIT"]]
    group_labels = ["Falter", "Non-Falter"]
    fig = ff.create_distplot(hist_data, group_labels,show_hist=False)
    fig['layout'].update(title={"text": 'Distribution of clients'}, xaxis_title="Amount of credit", yaxis_title="density")
    fig.add_vline(x=ccredit, line_width=1, line_color="black")
    st.plotly_chart(fig, use_container_width=True)

if selected_status == 'AMT_INCOME_TOTAL':
    hist_data = [train_df[train_df["TARGET"] == 1]["AMT_INCOME_TOTAL"], train_df[train_df["TARGET"] == 0]["AMT_INCOME_TOTAL"]]
    group_labels = ["Falter", "Non-Falter"]
    fig = ff.create_distplot(hist_data, group_labels,show_hist=False)
    fig['layout'].update(title={"text": 'Distribution of clients'}, xaxis_title="Income", yaxis_title="density")
    fig.add_vline(x=cincome, line_width=1, line_color="black")
    st.plotly_chart(fig, use_container_width=True)

if selected_status == 'DAYS_EMPLOYED':
    hist_data = [train_df[train_df["TARGET"] == 1]["DAYS_EMPLOYED"], train_df[train_df["TARGET"] == 0]["DAYS_EMPLOYED"]]
    group_labels = ["Falter", "Non-Falter"]
    fig = ff.create_distplot(hist_data, group_labels,show_hist=False)
    fig['layout'].update(title={"text": 'Distribution of clients'}, xaxis_title="Days employed", yaxis_title="density")
    fig.add_vline(x=cemployed, line_width=1, line_color="black")
    st.plotly_chart(fig, use_container_width=True)

st.title('Prediction')
rn=int((train_df[train_df['Client']==client_number].index).values)
train_df=train_df.drop(['Unnamed: 0'],axis=1)
train_df=train_df.drop(['Client'],axis=1)
X,y = (train_df.drop(['TARGET'],axis=1).values,train_df.TARGET.values)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)
rf = RandomForestClassifier(max_depth=4 , random_state=0,class_weight='balanced')
model=rf.fit(X_train, y_train)

y_pred2 = rf.predict(X)
score = y_pred2[rn]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

train_df=train_df.drop(['TARGET'],axis=1)
#lcol=train_df.columns
lcol=['Age','Employment','Id publish','Region','Commuting','Source 2','Phone','Gender','Working','Education','Credit','Income','Annuity']

if score==0:
    st.write('Predicted: Non-Falter')
if score==1:
    st.write('Predicted: Falter')

st.set_option('deprecation.showPyplotGlobalUse', False)
X_idx =rn
import streamlit.components.v1 as components
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

shap_value_single = explainer.shap_values(X = X[X_idx:X_idx+1,:])
plt.title('Feature importance based on SHAP values')
st_shap(shap.force_plot(base_value = explainer.expected_value[0],shap_values = shap_value_single[0],features = X[X_idx:X_idx+1,:],feature_names=lcol))
st.write('---')

#st.header('Feature Importance')
#plt.title('Feature importance based on SHAP values')
#shap.summary_plot(shap_values[0], X,feature_names=lcol)
#st.pyplot(bbox_inches='tight')
#st.write('---')



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
        self.df = pd.read_csv('/Users/riina//Downloads/finalcredit_df.csv')
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
