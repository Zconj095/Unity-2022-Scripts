import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
 
df = pd.read_csv('transcount.csv') 
df = df.groupby('year').aggregate(np.mean) 
years = df.index.values 
counts = df['trans_count'].values 
poly = np.polyfit(years, np.log(counts), deg=1) 
print("Poly", poly) 
plt.semilogy(years, counts, 'o') 
plt.semilogy(years, np.exp(np.polyval(poly, years))) 
plt.show()

poly = np.polyfit(years, np.log(counts), deg=1) 
print("Poly", poly) 

plt.semilogy(years, counts, 'o') 
plt.semilogy(years, np.exp(np.polyval(poly, years))) 






plt.scatter(years, cnt_log, c= 200 * years, s=20 + 200 * gpu_counts/gpu_counts.max(), alpha=0.5) 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
 
df = pd.read_csv('transcount.csv') 
df = df.groupby('year').aggregate(np.mean) 
 
gpu = pd.read_csv('gpu_transcount.csv') 
gpu = gpu.groupby('year').aggregate(np.mean) 
 
df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True) 
df = df.replace(np.nan, 0) 
print(df) 
years = df.index.values 
counts = df['trans_count'].values 
gpu_counts = df['gpu_trans_count'].values 
cnt_log = np.log(counts) 
plt.scatter(years, cnt_log, c= 200 * years, s=20 + 200 * gpu_counts/gpu_counts.max(), alpha=0.5) 
plt.show() 



plt.plot(years, np.polyval(poly, years), label='Fit') 
plt.scatter(years, cnt_log, c= 200 * years, s=20 + 200 * gpu_counts/gpu_counts.max(), alpha=0.5, label="Scatter Plot") 

gpu_start = gpu.index.values.min() 
y_ann = np.log(df.at[gpu_start, 'trans_count']) 
ann_str = "First GPU\n %d" % gpu_start 
plt.annotate(ann_str, xy=(gpu_start, y_ann), arrowprops=dict(arrowstyle="->"), xytext=(-30, +70), textcoords='offset points') 

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
 
df = pd.read_csv('transcount.csv') 
df = df.groupby('year').aggregate(np.mean) 
gpu = pd.read_csv('gpu_transcount.csv') 
gpu = gpu.groupby('year').aggregate(np.mean) 
 
df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True) 
df = df.replace(np.nan, 0) 
years = df.index.values 
counts = df['trans_count'].values 
gpu_counts = df['gpu_trans_count'].values 
 
poly = np.polyfit(years, np.log(counts), deg=1) 
plt.plot(years, np.polyval(poly, years), label='Fit') 
 
gpu_start = gpu.index.values.min() 
y_ann = np.log(df.at[gpu_start, 'trans_count']) 
ann_str = "First GPU\n %d" % gpu_start 
plt.annotate(ann_str, xy=(gpu_start, y_ann), arrowprops=dict(arrowstyle="->"), xytext=(-30, +70), textcoords='offset points') 
 
cnt_log = np.log(counts) 
plt.scatter(years, cnt_log, c= 200 * years, s=20 + 200 * gpu_counts/gpu_counts.max(), alpha=0.5, label="Scatter Plot") 
plt.legend(loc='upper left') 
plt.grid() 
plt.xlabel("Year") 
plt.ylabel("Log Transistor Counts", fontsize=16) 
plt.title("Moore's Law & Transistor Counts") 
plt.show() 



fig = plt.figure() 
ax = Axes3D(fig) 
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, Z)
ax.set_xlabel('Year') 
ax.set_ylabel('Log CPU transistor counts') 
ax.set_zlabel('Log GPU transistor counts') 
ax.set_title("Moore's Law & Transistor Counts")

from mpl_toolkits.mplot3d.axes3d import Axes3D 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
 
 
df = pd.read_csv('transcount.csv') 
df = df.groupby('year').aggregate(np.mean) 
gpu = pd.read_csv('gpu_transcount.csv') 
gpu = gpu.groupby('year').aggregate(np.mean) 
 
df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True) 
df = df.replace(np.nan, 0) 
 
fig = plt.figure() 
ax = Axes3D(fig) 
X = df.index.values 
Y = np.where(df['trans_count'].values>0,np.ma.log(df['trans_count'].values), 0) 
X, Y = np.meshgrid(X, Y) 
Z = np.where(df['gpu_trans_count'].values>0,np.ma.log(df['gpu_trans_count'].values), 0) 
ax.plot_surface(X, Y, Z) 
ax.set_xlabel('Year') 
ax.set_ylabel('Log CPU transistor counts') 
ax.set_zlabel('Log GPU transistor counts') 
ax.set_title("Moore's Law & Transistor Counts") 
plt.show()

df.plot(logy=True) 
df[df['gpu_trans_count'] > 0].plot(kind='scatter', x='trans_count', y='gpu_trans_count', loglog=True) 

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
 
df = pd.read_csv('transcount.csv') 
df = df.groupby('year').aggregate(np.mean) 
 
gpu = pd.read_csv('gpu_transcount.csv') 
gpu = gpu.groupby('year').aggregate(np.mean) 
 
df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True) 
df = df.replace(np.nan, 0) 
df.plot() 
df.plot(logy=True) 
df[df['gpu_trans_count'] > 0].plot(kind='scatter', x='trans_count', y='gpu_trans_count', loglog=True) 
plt.show() 




lag_plot(np.log(df['trans_count'])) 

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from pandas.tools.plotting import lag_plot 
 
df = pd.read_csv('transcount.csv') 
df = df.groupby('year').aggregate(np.mean) 
 
gpu = pd.read_csv('gpu_transcount.csv') 
gpu = gpu.groupby('year').aggregate(np.mean) 
 
df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True) 
df = df.replace(np.nan, 0) 
lag_plot(np.log(df['trans_count'])) 
plt.show() 


import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from pandas.tools.plotting import autocorrelation_plot 
 
df = pd.read_csv('transcount.csv') 
df = df.groupby('year').aggregate(np.mean) 
 
gpu = pd.read_csv('gpu_transcount.csv') 
gpu = gpu.groupby('year').aggregate(np.mean) 
 
df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True) 
df = df.replace(np.nan, 0) 
autocorrelation_plot(np.log(df['trans_count'])) 
plt.show() 

autocorrelation_plot(np.log(df['trans_count'])) 

pip install plotly
py.sign_in('username', 'api_key')
data = Data([Box(y=counts), Box(y=gpu_counts)])
plot_url = py.plot(data, filename='moore-law-scatter')

import plotly.plotly as py
from plotly.graph_objs import *
import numpy as np
import pandas as pd

df = pd.read_csv('transcount.csv')
df = df.groupby('year').aggregate(np.mean)

gpu = pd.read_csv('gpu_transcount.csv')
gpu = gpu.groupby('year').aggregate(np.mean)
df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True)
df = df.replace(np.nan, 0)

# Change the user and api_key to your own username and api_key
py.sign_in('username', 'api_key')

counts = np.log(df['trans_count'].values)
gpu_counts = np.log(df['gpu_trans_count'].values)

data = Data([Box(y=counts), Box(y=gpu_counts)])
plot_url = py.plot(data, filename='moore-law-scatter')
print(plot_url)


