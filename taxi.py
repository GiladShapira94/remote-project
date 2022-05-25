#!/usr/bin/env python
# coding: utf-8

# In[11]:


from os import path
import numpy as np 
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from pickle import dumps
import shapely.wkt


# In[12]:


def get_zones_dict(zones_url):
    zones_df = pd.read_csv(zones_url)
    
    # Remove unecessary fields
    zones_df.drop(['Shape_Leng', 'Shape_Area', 'zone', 'LocationID', 'borough'], axis=1, inplace=True)
    
    # Convert DF to dictionary
    zones_dict = zones_df.set_index('OBJECTID').to_dict('index')
    
    # Add lat/long to each zone
    for zone in zones_dict:
        shape = shapely.wkt.loads(zones_dict[zone]['the_geom'])
        zones_dict[zone]['long'] = shape.centroid.x
        zones_dict[zone]['lat'] = shape.centroid.y
    
    return zones_dict


# In[13]:


def get_zone_lat(zones_dict, zone_id):
    return zones_dict[zone_id]['lat']


# In[14]:


def get_zone_long(zones_dict, zone_id):
    return zones_dict[zone_id]['long']


# In[15]:


def clean_df(df):
    return df[(df.fare_amount > 0)  & (df.fare_amount <= 500) &
             (df.PULocationID > 0) & (df.PULocationID <= 263) & 
             (df.DOLocationID > 0) & (df.DOLocationID <= 263)]


# In[16]:


# To Compute Haversine distance
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    return 2 * R_earth * np.arcsin(np.sqrt(a))


# In[17]:


def radian_conv(degree):
    """
    Return radian.
    """
    return  np.radians(degree)


# In[18]:


def add_airport_dist(dataset):
    """
    Return minumum distance from pickup or dropoff coordinates to each airport.
    JFK: John F. Kennedy International Airport
    EWR: Newark Liberty International Airport
    LGA: LaGuardia Airport
    SOL: Statue of Liberty 
    NYC: Newyork Central
    """
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892,-74.0445) # Statue of Liberty
    nyc_coord = (40.7141667,-74.0063889) 
    
    
    pickup_lat = dataset['pickup_latitude']
    dropoff_lat = dataset['dropoff_latitude']
    pickup_lon = dataset['pickup_longitude']
    dropoff_lon = dataset['dropoff_longitude']
    
    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)
    pickup_sol = sphere_dist(pickup_lat, pickup_lon, sol_coord[0], sol_coord[1]) 
    dropoff_sol = sphere_dist(sol_coord[0], sol_coord[1], dropoff_lat, dropoff_lon)
    pickup_nyc = sphere_dist(pickup_lat, pickup_lon, nyc_coord[0], nyc_coord[1]) 
    dropoff_nyc = sphere_dist(nyc_coord[0], nyc_coord[1], dropoff_lat, dropoff_lon)
    
    
    
    dataset['jfk_dist'] = pickup_jfk + dropoff_jfk
    dataset['ewr_dist'] = pickup_ewr + dropoff_ewr
    dataset['lga_dist'] = pickup_lga + dropoff_lga
    dataset['sol_dist'] = pickup_sol + dropoff_sol
    dataset['nyc_dist'] = pickup_nyc + dropoff_nyc
    
    return dataset


# In[19]:


def add_datetime_info(dataset):
    #Convert to datetime format
    dataset['pickup_datetime'] = pd.to_datetime(dataset['tpep_pickup_datetime'],format="%Y-%m-%d %H:%M:%S")
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    dataset['year'] = dataset.pickup_datetime.dt.year
    
    return dataset


# In[20]:


def fetch_data(context : MLClientCtx, taxi_records_csv_path: DataItem, zones_csv_path: DataItem):
    context.logger.info('Reading taxi records data from {}'.format(taxi_records_csv_path))
    taxi_records_dataset = taxi_records_csv_path.as_df()
    
    context.logger.info('Reading zones data from {}'.format(zones_csv_path))
    zones_dataset = zones_csv_path.as_df()
    
    target_path = path.join(context.artifact_path)
    context.logger.info('Saving datasets to {} ...'.format(target_path))

    # Store the data sets in your artifacts database
    context.log_dataset('nyc-taxi-dataset', df=taxi_records_dataset, format='csv',
                        index=False, artifact_path=target_path)
    context.log_dataset('zones-dataset', df=zones_dataset, format='csv',
                        index=False, artifact_path=target_path)    


# In[21]:


def get_zones_dict(zones_df):

    # Remove unecessary fields
    zones_df.drop(['Shape_Leng', 'Shape_Area', 'zone', 'LocationID', 'borough'], axis=1, inplace=True)
    
    # Convert DF to dictionary
    zones_dict = zones_df.set_index('OBJECTID').to_dict('index')
    
    # Add lat/long to each zone
    for zone in zones_dict:
        shape = shapely.wkt.loads(zones_dict[zone]['the_geom'])
        zones_dict[zone]['long'] = shape.centroid.x
        zones_dict[zone]['lat'] = shape.centroid.y
    
    return zones_dict


# In[22]:


def get_zone_lat(zones_dict, zone_id):
    return zones_dict[zone_id]['lat']


# In[23]:


def get_zone_long(zones_dict, zone_id):
    return zones_dict[zone_id]['long']


# In[24]:


def transform_dataset(context : MLClientCtx, taxi_records_csv_path: DataItem, zones_csv_path: DataItem):
    
    context.logger.info('Begin datasets transform')
    
    context.logger.info('zones_csv_path: ' + str(zones_csv_path))
    
    zones_df = zones_csv_path.as_df()    
    
    # Get zones dictionary
    zones_dict = get_zones_dict(zones_df)
    
    train_df = taxi_records_csv_path.as_df()
    
    # Clean DF
    train_df = clean_df(train_df)
    
    # Enrich DF
    train_df['pickup_latitude'] = train_df.apply(lambda x: get_zone_lat(zones_dict, x['PULocationID']), axis=1 )
    train_df['pickup_longitude'] = train_df.apply(lambda x: get_zone_long(zones_dict, x['PULocationID']), axis=1 )
    train_df['dropoff_latitude'] = train_df.apply(lambda x: get_zone_lat(zones_dict, x['DOLocationID']), axis=1 )
    train_df['dropoff_longitude'] = train_df.apply(lambda x: get_zone_long(zones_dict, x['DOLocationID']), axis=1 )

    train_df = add_datetime_info(train_df)
    train_df = add_airport_dist(train_df)

    train_df['pickup_latitude'] = radian_conv(train_df['pickup_latitude'])
    train_df['pickup_longitude'] = radian_conv(train_df['pickup_longitude'])
    train_df['dropoff_latitude'] = radian_conv(train_df['dropoff_latitude'])
    train_df['dropoff_longitude'] = radian_conv(train_df['dropoff_longitude'])

    train_df.drop(['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'congestion_surcharge', 'improvement_surcharge', 'pickup_datetime',
                  'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'total_amount', 'RatecodeID', 'store_and_fwd_flag',
                  'PULocationID', 'DOLocationID', 'payment_type'], 
                  axis=1, inplace=True, errors='ignore')
    
    # Save dataset to artifact
    target_path = path.join(context.artifact_path)
    context.log_dataset('nyc-taxi-dataset-transformed', df=train_df, artifact_path=target_path, format='csv')    
    
    context.logger.info('End dataset transform')


# In[25]:


params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': 4,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,
        'num_rounds':50000
    }


# In[26]:


def train_model(context: MLClientCtx, input_ds: DataItem):
    
    context.logger.info('Begin training')
    context.logger.info('LGBM version is ' + str(lgbm.__version__))
    
    train_df = input_ds.as_df()
    
    y = train_df['fare_amount']
  
    train_df = train_df.drop(columns=['fare_amount'])
    train_df = train_df.drop(train_df.columns[[0]], axis=1)
    x_train,x_test,y_train,y_test = train_test_split(train_df,y,random_state=123,test_size=0.10)
    
    train_set = lgbm.Dataset(x_train, y_train, silent=False,categorical_feature=['year','month','day','weekday'])
    valid_set = lgbm.Dataset(x_test, y_test, silent=False,categorical_feature=['year','month','day','weekday'])
    model = lgbm.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=500,verbose_eval=500, valid_sets=valid_set)
    
    context.log_model('FareModel',
                     body=dumps(model),
                     artifact_path='s3://mlrun-v1-warroom/',
                     model_file="FareModel.pkl")
    
    context.logger.info('End training')

