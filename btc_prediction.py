#!/usr/bin/env python
# coding: utf-8

# In[15]:


import investpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import schedule
import time 
from nbloader import Notebook


# In[ ]:





# In[ ]:


schedule.every(1).hour.do()


# In[ ]:


df = investpy.get_crypto_historical_data(crypto='bitcoin', from_date='17/12/2019', to_date='30/03/2020')
df = pd.DataFrame(df)


# In[ ]:


df = df.drop(labels="Currency", axis = 1)
df.tail(30)


# In[ ]:


dfreg = df.loc[:,['Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0


# In[ ]:


import math
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


# In[ ]:


# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
# Separating the label here, we want to predict the Close
forecast_col = 'Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


# Linear regression


# In[ ]:


clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)


# In[ ]:


confidencereg = clfreg.score(X_test, y_test)


# In[ ]:


confidencereg


# In[ ]:


forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan


# In[ ]:


last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

    
dfreg['Close'].tail(10).plot()
dfreg['Forecast'].tail(10).plot()



plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[ ]:


dfreg.tail(50)

