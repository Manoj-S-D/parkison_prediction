from flask import Flask, request, jsonify, render_template, redirect, flash, send_file, Response
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore") 

mpl.rcParams["figure.figsize"] = [7, 7]
mpl.rcParams["figure.autolayout"] = True

app = Flask(__name__)

path = "Parkinsson_disease.csv"
df = pd.read_csv(path)
df.rename(columns=({
    'MDVP:Fo(Hz)':'avg_fre', 'MDVP:Fhi(Hz)':'max_fre', 'MDVP:Flo(Hz)':'min_fre', 'MDVP:Jitter(%)':'var_fre1',
    'MDVP:Jitter(Abs)':'var_fre2', 'MDVP:RAP':'var_fre3', 'MDVP:PPQ':'var_fre4', 'Jitter:DDP':'var_fre5',
    'MDVP:Shimmer':'var_amp1', 'MDVP:Shimmer(dB)':'var_amp2', 'Shimmer:APQ3':'var_amp3', 'Shimmer:APQ5':'var_amp4',
    'MDVP:APQ':'var_amp5', 'Shimmer:DDA':'var_amp6'
}), inplace=True)
df.drop(columns="name", axis=1, inplace=True)

x = df.loc[:, df.columns != 'status'].values[:, 1:]
x1 = df.loc[:, df.columns != 'status']
y = df.loc[:, 'status'].values
y1 = df.loc[:, 'status']
scaler = MinMaxScaler((-1, 1))
x1 = scaler.fit_transform(x)
y1 = y

xtrain, xtest, ytrain, ytest = train_test_split(x1, y1, test_size=0.2)
model = XGBClassifier()
model.fit(xtrain, ytrain)
predict = model.predict(xtest)
cm = confusion_matrix(ytest, predict)

plt.figure(figsize=(8, 6))
fg = sns.heatmap(cm, annot=True, cmap="Reds")
figure = fg.get_figure()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Output Confusion Matrix")
plt.savefig('static/images/confmat.png')

def makeplot(df=df):
    i = 0
    pd.DataFrame(x1).hist(figsize=(14, 14))
    i += 1
    plt.savefig(f'static/images/try{i}.png')

    df = df[df.max_fre <= 300]
    df = df[df.var_fre1 <= 0.02]
    df = df[df.var_fre2 <= 0.0001]
    df = df[df.var_fre3 <= 0.01]
    df = df[df.var_fre4 <= 0.01]
    df = df[df.var_fre5 <= 0.02]
    df = df[df.var_amp1 <= 0.10]
    df = df[df.var_amp2 <= 1.0]
    df = df[df.var_amp3 <= 0.04]
    df = df[df.var_amp4 <= 0.050]
    df = df[df.var_amp5 <= 0.075]
    df = df[df.var_amp6 <= 0.125]
    df = df[df.NHR <= 0.15]

    # Convert the column to a numpy array
    nhr_array = df['NHR'].values
    sns.histplot(nhr_array, kde=True)  # Use histplot with KDE overlay
    i += 1
    plt.savefig(f'static/images/try{i}.png')

    df = df[df.NHR <= 0.06]
    correl = pd.DataFrame(x1).corr()
    plt.figure(figsize=(15, 15))
    sns.heatmap(correl, cmap='OrRd', annot=True)
    i += 1
    plt.savefig(f'static/images/try{i}.png')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html', destab=[df.describe().to_html(classes="rep_tab")])

@app.route('/predict', methods=['POST'])
def predict():
    feature = [int(x) for x in request.form.values()]
    out = model.predict(np.array([feature]))
    return render_template('prediction.html', x=round(out[0]))

if __name__ == '__main__':
    makeplot()
    app.run(debug=True)
