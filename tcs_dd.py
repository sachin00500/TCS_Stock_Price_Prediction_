#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow
import keras
import streamlit as st
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from streamlit_option_menu import option_menu


# In[2]:


model=tensorflow.keras.models.load_model("C:/Users/LENOVO/Documents/model dd/tcs_model2.h5")


# In[3]:


def create_df(dataset,step):
    xxtrain,yytrain=[],[]
    for i in range(len(dataset)-step-1):
        a=dataset[i:(i+step),0]
        xxtrain.append(a)
        yytrain.append(dataset[i+step,0])
    return np.array(xxtrain),np.array(yytrain)


# In[4]:


options = option_menu("Main Menu",["Home"], icons=['house','gear-fill',"envelope"], menu_icon="cast", default_index=0,orientation="horizontal")
st.title("STOCK MARKET FORECASTING")


# In[5]:


a=st.sidebar.selectbox("STOCKS",("Select the stock","TCS"))


# In[6]:


df=pd.read_csv("C:/Users/LENOVO/Documents/excelr projects/forecasting 1st project/TCS.NS.csv")
st.dataframe(df)


# In[7]:


# Plotting Close Pric
st.subheader("closing price")
fig3=plt.figure(figsize=(12,6))
plt.plot(df.Close)


# In[8]:


# Plotting Close Price with MA100
st.subheader("closing price with 100MA")
ma1_100=df.Close.rolling(100).mean()
plt.plot(ma1_100)
st.pyplot(fig3)


# In[9]:


df_=pd.read_csv("C:/Users/LENOVO/Documents/excelr projects/forecasting 1st project/TCS.NS.csv")
df_=df_["Close"]


# In[10]:


# Performing LOG & SCALING 
df_log=np.log(df_)
normalizing=MinMaxScaler(feature_range=(0,1))
df_norm=normalizing.fit_transform(np.array(df_log).reshape(-1,1))


# In[11]:


t_s=100                            
df_x,df_y=create_df(df_norm, t_s)                                
fut_inp=df_y[2267:]
fut_inp=fut_inp.reshape(1,-1)   
temp_inp=list(fut_inp) 
temp_inp=temp_inp[0].tolist()


# In[12]:


fut_inp.shape


# In[13]:



lst_out=[]   
n_steps=100
i=1


# In[14]:


int_val = st.number_input('Seconds', min_value=1, max_value=10, value=5, step=1)


# In[15]:



    while(i<int_val):
        if(len(temp_inp)>100):
            fut_inp=np.array(temp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp=fut_inp.reshape((1,n_steps,1))
            yhat=model.predict(fut_inp,verbose=0)
            temp_inp.extend(yhat[0].tolist())
            temp_inp=temp_inp[1:]
            lst_out.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp=fut_inp.reshape((1,n_steps,1))
            yhat=model.predict(fut_inp,verbose=0)
            temp_inp.extend(yhat[0].tolist())
            lst_out.extend(yhat.tolist())
            i=i+1



# In[16]:



lst_out=normalizing.inverse_transform(lst_out)
lst_out=np.exp(lst_out) 
st.dataframe(lst_out)


# In[17]:


c=lst_out
fig4=plt.figure(figsize=(12,6))
plt.plot(c)
st.pyplot(fig4)


# In[ ]:




