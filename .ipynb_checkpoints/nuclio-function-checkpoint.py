#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd

def handler(context,event):
    df = pd.read_csv('s3://mlrun-v1-warroom/nyc-taxi-dataset-transformed.csv')
    lasr_drive = df[['hour','day','month','year']].sort_values(by=['year', 'month', 'day', 'hour'],ascending=False).head(1).to_dict(orient='list')
    return lasr_drive


# In[ ]:




