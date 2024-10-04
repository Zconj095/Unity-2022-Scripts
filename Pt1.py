import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from pandas.stats.moments import rolling_mean 
 
data_loader = sm.datasets.sunspots.load_pandas() 
df = data_loader.data 
year_range = df["YEAR"].values 
plt.plot(year_range, df["SUNACTIVITY"].values, label="Original") 
plt.plot(year_range, df.rolling(window=11).mean()["SUNACTIVITY"].values, label="SMA 11") 
plt.plot(year_range, df.rolling(window=22).mean()["SUNACTIVITY"].values, label="SMA 22") 
plt.legend() 
plt.show() 

weights = np.exp(np.linspace(-1., 0., N)) 
weights /= weights.sum() 

def sma(arr, n): 
   weights = np.ones(n) / n 
 
   return np.convolve(weights, arr)[n-1:-n+1]  

data_loader = sm.datasets.sunspots.load_pandas() 
df = data_loader.data 

pip install statsmodels

w(n) = 1 

import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from pandas.stats.moments import rolling_window 
import pandas as pd 
 
data_loader = sm.datasets.sunspots.load_pandas() 
df = data_loader.data.tail(150) 
df = pd.DataFrame({'SUNACTIVITY':df['SUNACTIVITY'].values}, index=df['YEAR']) 
ax = df.plot() 
 
def plot_window(wintype): 
    df2 = df.rolling(window=22,win_type=wintype, 
center=False,axis=0).mean() 
    df2.columns = [wintype] 
    df2.plot(ax=ax) 
 
plot_window('boxcar') 
plot_window('triang') 
plot_window('blackman') 
plot_window('hanning') 
plot_window('bartlett') 
plt.show() 

import statsmodels.api as sm 
from pandas.stats.moments import rolling_window 
import pandas as pd 
import statsmodels.tsa.stattools as ts 
import numpy as np 
 
def calc_adf(x, y): 
    result = sm.OLS(x, y).fit()     
    return ts.adfuller(result.resid) 
 
data_loader = sm.datasets.sunspots.load_pandas() 
data = data_loader.data.values 
N = len(data) 
 
t = np.linspace(-2 * np.pi, 2 * np.pi, N) 
sine = np.sin(np.sin(t)) 
print("Self ADF", calc_adf(sine, sine)) 
 
noise = np.random.normal(0, .01, N) 
print("ADF sine with noise", calc_adf(sine, sine + noise)) 
 
cosine = 100 * np.cos(t) + 10 
print("ADF sine vs cosine with noise", calc_adf(sine, cosine + noise)) 
 
print("Sine vs sunspots", calc_adf(sine, data)) 

def calc_adf(x, y): 
    result = sm.OLS(x, y).fit()     
    return ts.adfuller(result.resid) 

data_loader = sm.datasets.sunspots.load_pandas() 
data = data_loader.data.values 
N = len(data) 

t = np.linspace(-2 * np.pi, 2 * np.pi, N) 
sine = np.sin(np.sin(t)) 
print("Self ADF", calc_adf(sine, sine))

noise = np.random.normal(0, .01, N) 
print("ADF sine with noise", calc_adf(sine, sine + noise)) 

cosine = 100 * np.cos(t) + 10 
print("ADF sine vs cosine with noise", calc_adf(sine, cosine + noise)) 

Sine vs sunspots (-6.7242691810701016, 3.4210811915549028e-09, 16, 292, {'5%': -2.8714895534256861, '1%': -3.4529449243622383, '10%': -2.5720714378870331}, -1102.5867415291168)

y = data - np.mean(data) 
norm = np.sum(y ** 2) 
correlated = np.correlate(y, y, mode='full')/norm 

print np.argsort(res)[-5:] 

import numpy as np 
import pandas as pd 
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
from pandas import plotting
from pandas.tools.plotting import autocorrelation_plot

data_loader = sm.datasets.sunspots.load_pandas() 
data = data_loader.data["SUNACTIVITY"].values 
y = data - np.mean(data) 
norm = np.sum(y ** 2) 
correlated = np.correlate(y, y, mode='full')/norm 
res = correlated[len(correlated)/2:] 
 
print(np.argsort(res)[-5:]) 
plt.plot(res) 
plt.grid(True) 
plt.xlabel("Lag") 
plt.ylabel("Autocorrelation") 
plt.show() 
autocorrelation_plot(data) 
plt.show() 

def model(p, x1, x10): 
   p1, p10 = p 
   return p1 * x1 + p10 * x10 
 
def error(p, data, x1, x10): 
   return data - model(p, x1, x10) 

def fit(data): 
   p0 = [.5, 0.5] 
   params = leastsq(error, p0, args=(data[10:], data[9:-1], data[:-10]))[0] 
   return params 

cutoff = .9 * len(sunspots) 
params = fit(sunspots[:cutoff]) 
print "Params", params 

from scipy.optimize import leastsq 
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
import numpy as np 
 
def model(p, x1, x10): 
   p1, p10 = p 
   return p1 * x1 + p10 * x10 
 
def error(p, data, x1, x10): 
   return data - model(p, x1, x10) 
 
def fit(data): 
   p0 = [.5, 0.5] 
   params = leastsq(error, p0, args=(data[10:], data[9:-1], data[:-10]))[0] 
   return params 
 
data_loader = sm.datasets.sunspots.load_pandas() 
sunspots = data_loader.data["SUNACTIVITY"].values 
 
cutoff = .9 * len(sunspots) 
params = fit(sunspots[:cutoff]) 
print("Params", params) 
 
pred = params[0] * sunspots[cutoff-1:-1] + params[1] * sunspots[cutoff-10:-10] 
actual = sunspots[cutoff:] 
print("Root mean square error", np.sqrt(np.mean((actual - pred) ** 2))) 
print("Mean absolute error", np.mean(np.abs(actual - pred))) 
print("Mean absolute percentage error", 100 * np.mean(np.abs(actual - pred)/actual)) 
mid = (actual + pred)/2 
print("Symmetric Mean absolute percentage error", 100 * np.mean(np.abs(actual - pred)/mid)) 
print("Coefficient of determination", 1 - ((actual - pred) ** 2).sum()/ ((actual - actual.mean()) ** 2).sum()) 
year_range = data_loader.data["YEAR"].values[cutoff:] 
plt.plot(year_range, actual, 'o', label="Sunspots") 
plt.plot(year_range, pred, 'x', label="Prediction") 
plt.grid(True) 
plt.xlabel("YEAR") 
plt.ylabel("SUNACTIVITY") 
plt.legend() 
plt.show() 

model = sm.tsa.ARMA(df, (10,1)).fit() 

prediction = model.predict('1975', str(years[-1]), dynamic=True) 

import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
import datetime 
 
data_loader = sm.datasets.sunspots.load_pandas() 
df = data_loader.data 
years = df["YEAR"].values.astype(int) 
df.index = pd.Index(sm.tsa.datetools.dates_from_range(str(years[0]), str(years[-1]))) 
del df["YEAR"] 
 
model = sm.tsa.ARMA(df, (10,1)).fit() 
prediction = model.predict('1975', str(years[-1]), dynamic=True) 
 
df['1975':].plot() 
prediction.plot(style='--', label='Prediction') 
plt.legend() 
plt.show() 

from scipy.optimize import leastsq 
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
import numpy as np 
def model(p, t): 
   C, p1, f1, phi1 , p2, f2, phi2, p3, f3, phi3 = p 
   return C + p1 * np.sin(f1 * t + phi1) + p2 * np.sin(f2 * t + phi2) +p3 * np.sin(f3 * t + phi3) 
 
 
def error(p, y, t): 
   return y - model(p, t) 
 
def fit(y, t): 
   p0 = [y.mean(), 0, 2 * np.pi/11, 0, 0, 2 * np.pi/22, 0, 0, 2 * np.pi/100, 0] 
   params = leastsq(error, p0, args=(y, t))[0] 
   return params 
 
data_loader = sm.datasets.sunspots.load_pandas() 
sunspots = data_loader.data["SUNACTIVITY"].values 
years = data_loader.data["YEAR"].values 
 
cutoff = .9 * len(sunspots) 
params = fit(sunspots[:cutoff], years[:cutoff]) 
print("Params", params) 
 
pred = model(params, years[cutoff:]) 
actual = sunspots[cutoff:] 
print("Root mean square error", np.sqrt(np.mean((actual - pred) ** 2))) 
print("Mean absolute error", np.mean(np.abs(actual - pred))) 
print("Mean absolute percentage error", 100 * np.mean(np.abs(actual - pred)/actual)) 
mid = (actual + pred)/2 
print("Symmetric Mean absolute percentage error", 100 * np.mean(np.abs(actual - pred)/mid)) 
print("Coefficient of determination", 1 - ((actual - pred) ** 2).sum()/ ((actual - actual.mean()) ** 2).sum()) 
year_range = data_loader.data["YEAR"].values[cutoff:] 
plt.plot(year_range, actual, 'o', label="Sunspots") 
plt.plot(year_range, pred, 'x', label="Prediction") 
plt.grid(True) 
plt.xlabel("YEAR") 
plt.ylabel("SUNACTIVITY") 
plt.legend() 
plt.show() 

from scipy.fftpack import rfft 
from scipy.fftpack import fftshift 

t = np.linspace(-2 * np.pi, 2 * np.pi, len(sunspots)) 
mid = np.ptp(sunspots)/2 
sine = mid + mid * np.sin(np.sin(t)) 
 
sine_fft = np.abs(fftshift(rfft(sine))) 
print "Index of max sine FFT", np.argsort(sine_fft)[-5:] 

transformed = np.abs(fftshift(rfft(sunspots))) 
print "Indices of max sunspots FFT", np.argsort(transformed)[-5:] 

import numpy as np 
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
from scipy.fftpack import rfft 
from scipy.fftpack import fftshift 
 
data_loader = sm.datasets.sunspots.load_pandas() 
sunspots = data_loader.data["SUNACTIVITY"].values 
 
t = np.linspace(-2 * np.pi, 2 * np.pi, len(sunspots)) 
mid = np.ptp(sunspots)/2 
sine = mid + mid * np.sin(np.sin(t)) 
 
sine_fft = np.abs(fftshift(rfft(sine))) 
print("Index of max sine FFT", np.argsort(sine_fft)[-5:]) 
 
transformed = np.abs(fftshift(rfft(sunspots))) 
print("Indices of max sunspots FFT", np.argsort(transformed)[-5:]) 
 
plt.subplot(311) 
plt.plot(sunspots, label="Sunspots") 
plt.plot(sine, lw=2, label="Sine") 
plt.grid(True) 
plt.legend() 
plt.subplot(312) 
plt.plot(transformed, label="Transformed Sunspots") 
plt.grid(True) 
plt.legend() 
plt.subplot(313) 
plt.plot(sine_fft, lw=2, label="Transformed Sine") 
plt.grid(True) 
plt.legend() 
plt.show() 

plt.plot(transformed ** 2, label="Power Spectrum") 

plt.plot(np.angle(transformed), label="Phase Spectrum") 

import statsmodels.api as sm 
import matplotlib.pyplot as plt 
from scipy.signal import medfilt 
from scipy.signal import wiener 
from scipy.signal import detrend 
 
data_loader = sm.datasets.sunspots.load_pandas() 
sunspots = data_loader.data["SUNACTIVITY"].values 
years = data_loader.data["YEAR"].values 
 
plt.plot(years, sunspots, label="SUNACTIVITY") 
plt.plot(years, medfilt(sunspots, 11), lw=2, label="Median") 
plt.plot(years, wiener(sunspots, 11), '--', lw=2, label="Wiener") 
plt.plot(years, detrend(sunspots), lw=3, label="Detrend") 
plt.xlabel("YEAR") 
plt.grid(True) 
plt.legend() 
plt.show() 

scaled = preprocessing.scale(rain) 
binarized = preprocessing.binarize(rain) 
print(np.unique(binarized), binarized.sum()) 

lb = preprocessing.LabelBinarizer() 
lb.fit(rain.astype(int)) 
print(lb.classes_) 

import numpy as np 
from sklearn import preprocessing 
from scipy.stats import anderson 
 
rain = np.load('rain.npy') 
rain = .1 * rain 
rain[rain < 0] = .05/2 
print("Rain mean", rain.mean()) 
print("Rain variance", rain.var()) 
print("Anderson rain", anderson(rain)) 
 
scaled = preprocessing.scale(rain) 
print("Scaled mean", scaled.mean()) 
print("Scaled variance", scaled.var()) 
print("Anderson scaled", anderson(scaled)) 
 
binarized = preprocessing.binarize(rain) 
print(np.unique(binarized), binarized.sum()) 
 
lb = preprocessing.LabelBinarizer() 
lb.fit(rain.astype(int)) 
print(lb.classes_) 

clf = LogisticRegression(random_state=12) 
kf = KFold(len(y), n_folds=10) 
clf.fit(x[train], y[train]) 
scores.append(clf.score(x[test], y[test])) 
x = np.vstack((dates[:-1], rain[:-1])) 
y = np.sign(rain[1:]) 

from sklearn.linear_model import LogisticRegression 
from sklearn.cross_validation import KFold 
from sklearn import datasets 
import numpy as np 
 
def classify(x, y): 
    clf = LogisticRegression(random_state=12) 
    scores = [] 
    kf = KFold(len(y), n_folds=10) 
    for train,test in kf: 
      clf.fit(x[train], y[train]) 
      scores.append(clf.score(x[test], y[test])) 
 
    print("accuracy", np.mean(scores)) 
 
rain = np.load('rain.npy') 
dates = np.load('doy.npy') 
 
x = np.vstack((dates[:-1], rain[:-1])) 
y = np.sign(rain[1:]) 
classify(x.T, y) 
 
#iris example 
iris = datasets.load_iris() 
x = iris.data[:, :2] 
y = iris.target 
classify(x, y) 

clf = GridSearchCV(SVC(random_state=42, max_iter=100), {'kernel': ['linear', 'poly', 'rbf'], 'C':[1, 10]}) 

from sklearn.svm import SVC 
from sklearn.grid_search import GridSearchCV 
from sklearn import datasets 
import numpy as np 
from pprint import PrettyPrinter 
 
def classify(x, y): 
    clf = GridSearchCV(SVC(random_state=42, max_iter=100), {'kernel': ['linear', 'poly', 'rbf'], 'C':[1, 10]}) 
 
    clf.fit(x, y) 
    print("Score", clf.score(x, y)) 
    PrettyPrinter().pprint(clf.grid_scores_) 
 
rain = np.load('rain.npy') 
dates = np.load('doy.npy') 
 
x = np.vstack((dates[:-1], rain[:-1])) 
y = np.sign(rain[1:]) 
classify(x.T, y) 
 
#iris example 
iris = datasets.load_iris() 
x = iris.data[:, :2] 
y = iris.target 
classify(x, y) 

clf = ElasticNetCV(max_iter=200, cv=10, l1_ratio = [.1, .5, .7, .9, .95, .99, 1]) 

from sklearn.linear_model import ElasticNetCV 
import numpy as np 
from sklearn import datasets 
import matplotlib.pyplot as plt 
 
 
def regress(x, y, title): 
    clf = ElasticNetCV(max_iter=200, cv=10, l1_ratio = [.1, .5, .7, .9, .95, .99, 1]) 
 
    clf.fit(x, y) 
    print("Score", clf.score(x, y)) 
 
    pred = clf.predict(x) 
    plt.title("Scatter plot of prediction and " + title) 
    plt.xlabel("Prediction") 
    plt.ylabel("Target") 
    plt.scatter(y, pred) 
    # Show perfect fit line 
    if "Boston" in title: 
        plt.plot(y, y, label="Perfect Fit") 
        plt.legend() 
 
    plt.grid(True) 
    plt.show() 
 
rain = .1 * np.load('rain.npy') 
rain[rain < 0] = .05/2 
dates = np.load('doy.npy') 
 
x = np.vstack((dates[:-1], rain[:-1])) 
y = rain[1:] 
regress(x.T, y, "rain data") 
 
boston = datasets.load_boston() 
x = boston.data 
y = boston.target 
regress(x, y, "Boston house prices") 

train_sizes, train_scores, test_scores = learning_curve(clf, X, Y, n_jobs=ncpus) 

plt.plot(train_sizes, train_scores.mean(axis=1), label="Train score") 
plt.plot(train_sizes, test_scores.mean(axis=1), '--', label="Test score") 

import numpy as np 
from sklearn import datasets 
from sklearn.learning_curve import learning_curve 
from sklearn.svm import SVR 
from sklearn import preprocessing 
import multiprocessing 
import matplotlib.pyplot as plt 
 
 
def regress(x, y, ncpus, title): 
    X = preprocessing.scale(x) 
    Y = preprocessing.scale(y) 
    clf = SVR(max_iter=ncpus * 200) 
 
    train_sizes, train_scores, test_scores = learning_curve(clf, X, Y, n_jobs=ncpus)  
 
    plt.figure() 
    plt.title(title) 
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train score") 
    plt.plot(train_sizes, test_scores.mean(axis=1), '--', label="Test score") 
    print("Max test score " + title, test_scores.max()) 
    plt.grid(True) 
    plt.legend(loc='best') 
    plt.show() 
 
rain = .1 * np.load('rain.npy') 
rain[rain < 0] = .05/2 
dates = np.load('doy.npy') 
 
x = np.vstack((dates[:-1], rain[:-1])) 
y = rain[1:] 
ncpus = multiprocessing.cpu_count() 
regress(x.T, y, ncpus, "Rain") 
 
boston = datasets.load_boston() 
x = boston.data 
y = boston.target 
regress(x, y, ncpus, "Boston") 

x, _ = datasets.make_blobs(n_samples=100, centers=3, n_features=2, random_state=10) 

S = euclidean_distances(x) 

aff_pro = cluster.AffinityPropagation().fit(S) 
labels = aff_pro.labels_ 

from sklearn import datasets 
from sklearn import cluster 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import euclidean_distances 
 
 
x, _ = datasets.make_blobs(n_samples=100, centers=3, n_features=2, random_state=10) 
S = euclidean_distances(x) 
 
aff_pro = cluster.AffinityPropagation().fit(S) 
labels = aff_pro.labels_ 
 
styles = ['o', 'x', '^'] 
 
for style, label in zip(styles, np.unique(labels)): 
   print(label) 
   plt.plot(x[labels == label], style, label=label) 
plt.title("Clustering Blobs") 
plt.grid(True) 
plt.legend(loc='best') 
plt.show() 

df = pd.DataFrame.from_records(x.T, columns=['dates', 'rain']) 
df = df.groupby('dates').mean() 
 
df.plot() 

x = np.vstack((np.arange(1, len(df) + 1) , df.as_matrix().ravel())) 
x = x.T 
ms = cluster.MeanShift() 
ms.fit(x) 
labels = ms.predict(x) 

import numpy as np 
from sklearn import cluster 
import matplotlib.pyplot as plt 
import pandas as pd 
 
rain = .1 * np.load('rain.npy') 
rain[rain < 0] = .05/2 
dates = np.load('doy.npy') 
x = np.vstack((dates, rain)) 
df = pd.DataFrame.from_records(x.T, columns=['dates', 'rain']) 
df = df.groupby('dates').mean() 
df.plot() 
x = np.vstack((np.arange(1, len(df) + 1) , df.as_matrix().ravel())) 
x = x.T 
ms = cluster.MeanShift() 
ms.fit(x) 
labels = ms.predict(x) 
 
plt.figure() 
grays = ['0', '0.5', '0.75'] 
 
for gray, label in zip(grays, np.unique(labels)): 
    match = labels == label 
    x0 = x[:, 0] 
    x1 = x[:, 1] 
    plt.plot(x0[match], x1[match], lw=label+1, label=label) 
    plt.fill_between(x0, x1, where=match, color=gray) 
 
plt.grid(True) 
plt.legend() 
plt.show() 


creator.create("FitnessMax", base.Fitness, weights=(1.0,)) 
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax) 

toolbox = base.Toolbox() 
toolbox.register("attr_float", random.random) 
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 200) 
toolbox.register("populate", tools.initRepeat, list, toolbox.individual) 

def eval(individual): 
    return shapiro(individual)[1], 

toolbox.register("evaluate", eval) 
toolbox.register("mate", tools.cxTwoPoint) 
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1) 
toolbox.register("select", tools.selTournament, tournsize=4) 

pop = toolbox.populate(n=400) 

hof = tools.HallOfFame(1) 
stats = tools.Statistics(key=lambda ind: ind.fitness.values) 
stats.register("max", np.max) 
 
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=80, stats=stats, halloffame=hof) 

import array 
import random 
import numpy as np 
from deap import algorithms 
from deap import base 
from deap import creator 
from deap import tools 
from scipy.stats import shapiro 
import matplotlib.pyplot as plt 
 
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) 
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax) 
 
toolbox = base.Toolbox() 
toolbox.register("attr_float", random.random) 
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 200) 
toolbox.register("populate", tools.initRepeat, list, toolbox.individual) 
 
def eval(individual): 
    return shapiro(individual)[1], 
 
toolbox.register("evaluate", eval) 
toolbox.register("mate", tools.cxTwoPoint) 
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1) 
toolbox.register("select", tools.selTournament, tournsize=4) 
 
random.seed(42) 
 
pop = toolbox.populate(n=400) 
hof = tools.HallOfFame(1) 
stats = tools.Statistics(key=lambda ind: ind.fitness.values) 
stats.register("max", np.max) 
 
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=80, stats=stats, halloffame=hof) 
 
print(shapiro(hof[0])[1]) 
plt.hist(hof[0]) 
plt.grid(True) 
plt.show() 

pip install theanets nose_parameterized
net = theanets.Regressor(layers=[2,3,1]) 
train = [x[:N], y[:N]] 
valid = [x[N:], y[N:]] 
net.train(train,valid,learning_rate=0.1,momentum=0.5) 
pred = net.predict(x[N:]).ravel() 
print("Pred Min", pred.min(), "Max", pred.max()) 
print("Y Min", y.min(), "Max", y.max()) 
print("Accuracy", accuracy_score(y[N:], pred >= .5)) 

import numpy as np 
import theanets 
import multiprocessing 
from sklearn import datasets 
from sklearn.metrics import accuracy_score 
 
rain = .1 * np.load('rain.npy') 
rain[rain < 0] = .05/2 
dates = np.load('doy.npy') 
x = np.vstack((dates[:-1], np.sign(rain[:-1]))) 
x = x.T 
 
y = np.vstack(np.sign(rain[1:]),) 
N = int(.9 * len(x)) 
 
train = [x[:N], y[:N]] 
valid = [x[N:], y[N:]] 
 
net = theanets.Regressor(layers=[2,3,1]) 
 
net.train(train,valid,learning_rate=0.1,momentum=0.5) 
 
pred = net.predict(x[N:]).ravel() 
print("Pred Min", pred.min(), "Max", pred.max()) 
print("Y Min", y.min(), "Max", y.max()) 
print("Accuracy", accuracy_score(y[N:], pred >= .5)) 


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=37) 
clf = tree.DecisionTreeClassifier(random_state=37) 

params = {"max_depth": [2, None], 
              "min_samples_leaf": sp_randint(1, 5), 
              "criterion": ["gini", "entropy"]} 
rscv = RandomizedSearchCV(clf, params) 
rscv.fit(x_train,y_train) 

sio = io.StringIO() 
tree.export_graphviz(rscv.best_estimator_, out_file=sio, feature_names=['day-of-year','yest']) 
dec_tree = pydot.graph_from_dot_data(sio.getvalue()) 
 
print("Best Train Score", rscv.best_score_) 
print("Test Score", rscv.score(x_test, y_test)) 
print("Best params", rscv.best_params_) 
 
from IPython.display import Image 
Image(dec_tree.create_png()) 

from sklearn.model_selection import train_test_split 
from sklearn import tree 
from sklearn.model_selection import RandomizedSearchCV 
from scipy.stats import randint as sp_randint 
import pydotplus as pydot 
import io 
import numpy as np 
 
rain = .1 * np.load('rain.npy') 
rain[rain < 0] = .05/2 
dates = np.load('doy.npy').astype(int) 
x = np.vstack((dates[:-1], np.sign(rain[:-1]))) 
x = x.T 
 
y = np.sign(rain[1:]) 
 
x_train, x_test, y_train, y_test = train_test_split(x, y,  
random_state=37) 
 
clf = tree.DecisionTreeClassifier(random_state=37) 
params = {"max_depth": [2, None], 
              "min_samples_leaf": sp_randint(1, 5), 
              "criterion": ["gini", "entropy"]} 
rscv = RandomizedSearchCV(clf, params) 
rscv.fit(x_train,y_train) 
 
sio = io.StringIO() 
tree.export_graphviz(rscv.best_estimator_, out_file=sio,  
feature_names=['day-of-year','yest']) 
dec_tree = pydot.graph_from_dot_data(sio.getvalue()) 
 
print("Best Train Score", rscv.best_score_) 
print("Test Score", rscv.score(x_test, y_test)) 
print("Best params", rscv.best_params_) 
 
from IPython.display import Image 
Image(dec_tree.create_png()) 

