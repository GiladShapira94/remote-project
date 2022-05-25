#!/usr/bin/env python
# coding: utf-8

# # Model Serving Function

# In[1]:


import mlrun
import os


# In[2]:


import numpy as np
from cloudpickle import load

class LGBMModel(mlrun.serving.V2ModelServer):
    
    def load(self):
        model_file, extra_data = self.get_model('.pkl')
        self.model = load(open(model_file, 'rb'))

    def predict(self, body):
        try:
            feats = np.asarray(body['inputs'])
            result = self.model.predict(feats)
            return result.tolist()
        except Exception as e:
            raise Exception("Failed to predict %s" % e)


# In[3]:


# nuclio: end-code


# ## Deploy and Test The Function

# This demo uses a Model file from MLRUn demo data repository(by default stored in Wasabi object-store service).

# In[4]:


models_path = mlrun.get_sample_path('models/lightgbm/SampleModel.pkl')


# In[5]:


fn = mlrun.code_to_function('lightgbm-serving',
                            description="LightGBM Serving",
                            categories=['serving', 'ml'],
                            labels={'author': 'edmondg', 'framework': 'lightgbm'},
                            code_output='.',
                            image='mlrun/mlrun',
                            kind='serving')
fn.spec.build.commands = ['pip install lightgbm']
fn.spec.default_class = 'LGBMModel'


# In[6]:


fn.add_model('nyc-taxi-server', model_path=models_path)


# In[7]:


# deploy the function
fn.apply(mlrun.platforms.auto_mount())
address = fn.deploy()


# In[8]:


# test the function
my_data = '''{"inputs":[[5.1, 3.5, 1.4, 3, 5.1, 3.5, 1.4, 0.2, 5.1, 3.5, 1.4, 0.2, 5.1, 3.5, 1.4, 0.2]]}'''
fn.invoke('/v2/models/nyc-taxi-server/predict', my_data)


# In[ ]:




