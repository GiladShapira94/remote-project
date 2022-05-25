{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "from os import path\n",
    "import numpy as np \n",
    "import pandas as pd\n",
    "import datetime as dt\n",
    "from sklearn.model_selection import train_test_split\n",
    "import lightgbm as lgbm\n",
    "from mlrun.execution import MLClientCtx\n",
    "from mlrun.datastore import DataItem\n",
    "from pickle import dumps\n",
    "import shapely.wkt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_zones_dict(zones_url):\n",
    "    zones_df = pd.read_csv(zones_url)\n",
    "    \n",
    "    # Remove unecessary fields\n",
    "    zones_df.drop(['Shape_Leng', 'Shape_Area', 'zone', 'LocationID', 'borough'], axis=1, inplace=True)\n",
    "    \n",
    "    # Convert DF to dictionary\n",
    "    zones_dict = zones_df.set_index('OBJECTID').to_dict('index')\n",
    "    \n",
    "    # Add lat/long to each zone\n",
    "    for zone in zones_dict:\n",
    "        shape = shapely.wkt.loads(zones_dict[zone]['the_geom'])\n",
    "        zones_dict[zone]['long'] = shape.centroid.x\n",
    "        zones_dict[zone]['lat'] = shape.centroid.y\n",
    "    \n",
    "    return zones_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_zone_lat(zones_dict, zone_id):\n",
    "    return zones_dict[zone_id]['lat']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_zone_long(zones_dict, zone_id):\n",
    "    return zones_dict[zone_id]['long']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_df(df):\n",
    "    return df[(df.fare_amount > 0)  & (df.fare_amount <= 500) &\n",
    "             (df.PULocationID > 0) & (df.PULocationID <= 263) & \n",
    "             (df.DOLocationID > 0) & (df.DOLocationID <= 263)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "# To Compute Haversine distance\n",
    "def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):\n",
    "    \"\"\"\n",
    "    Return distance along great radius between pickup and dropoff coordinates.\n",
    "    \"\"\"\n",
    "    #Define earth radius (km)\n",
    "    R_earth = 6371\n",
    "    #Convert degrees to radians\n",
    "    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,\n",
    "                                                             [pickup_lat, pickup_lon, \n",
    "                                                              dropoff_lat, dropoff_lon])\n",
    "    #Compute distances along lat, lon dimensions\n",
    "    dlat = dropoff_lat - pickup_lat\n",
    "    dlon = dropoff_lon - pickup_lon\n",
    "    \n",
    "    #Compute haversine distance\n",
    "    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2\n",
    "    return 2 * R_earth * np.arcsin(np.sqrt(a))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def radian_conv(degree):\n",
    "    \"\"\"\n",
    "    Return radian.\n",
    "    \"\"\"\n",
    "    return  np.radians(degree)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "def add_airport_dist(dataset):\n",
    "    \"\"\"\n",
    "    Return minumum distance from pickup or dropoff coordinates to each airport.\n",
    "    JFK: John F. Kennedy International Airport\n",
    "    EWR: Newark Liberty International Airport\n",
    "    LGA: LaGuardia Airport\n",
    "    SOL: Statue of Liberty \n",
    "    NYC: Newyork Central\n",
    "    \"\"\"\n",
    "    jfk_coord = (40.639722, -73.778889)\n",
    "    ewr_coord = (40.6925, -74.168611)\n",
    "    lga_coord = (40.77725, -73.872611)\n",
    "    sol_coord = (40.6892,-74.0445) # Statue of Liberty\n",
    "    nyc_coord = (40.7141667,-74.0063889) \n",
    "    \n",
    "    \n",
    "    pickup_lat = dataset['pickup_latitude']\n",
    "    dropoff_lat = dataset['dropoff_latitude']\n",
    "    pickup_lon = dataset['pickup_longitude']\n",
    "    dropoff_lon = dataset['dropoff_longitude']\n",
    "    \n",
    "    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) \n",
    "    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) \n",
    "    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])\n",
    "    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) \n",
    "    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) \n",
    "    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)\n",
    "    pickup_sol = sphere_dist(pickup_lat, pickup_lon, sol_coord[0], sol_coord[1]) \n",
    "    dropoff_sol = sphere_dist(sol_coord[0], sol_coord[1], dropoff_lat, dropoff_lon)\n",
    "    pickup_nyc = sphere_dist(pickup_lat, pickup_lon, nyc_coord[0], nyc_coord[1]) \n",
    "    dropoff_nyc = sphere_dist(nyc_coord[0], nyc_coord[1], dropoff_lat, dropoff_lon)\n",
    "    \n",
    "    \n",
    "    \n",
    "    dataset['jfk_dist'] = pickup_jfk + dropoff_jfk\n",
    "    dataset['ewr_dist'] = pickup_ewr + dropoff_ewr\n",
    "    dataset['lga_dist'] = pickup_lga + dropoff_lga\n",
    "    dataset['sol_dist'] = pickup_sol + dropoff_sol\n",
    "    dataset['nyc_dist'] = pickup_nyc + dropoff_nyc\n",
    "    \n",
    "    return dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "def add_datetime_info(dataset):\n",
    "    #Convert to datetime format\n",
    "    dataset['pickup_datetime'] = pd.to_datetime(dataset['tpep_pickup_datetime'],format=\"%Y-%m-%d %H:%M:%S\")\n",
    "    \n",
    "    dataset['hour'] = dataset.pickup_datetime.dt.hour\n",
    "    dataset['day'] = dataset.pickup_datetime.dt.day\n",
    "    dataset['month'] = dataset.pickup_datetime.dt.month\n",
    "    dataset['weekday'] = dataset.pickup_datetime.dt.weekday\n",
    "    dataset['year'] = dataset.pickup_datetime.dt.year\n",
    "    \n",
    "    return dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fetch_data(context : MLClientCtx, taxi_records_csv_path: DataItem, zones_csv_path: DataItem):\n",
    "    context.logger.info('Reading taxi records data from {}'.format(taxi_records_csv_path))\n",
    "    taxi_records_dataset = taxi_records_csv_path.as_df()\n",
    "    \n",
    "    context.logger.info('Reading zones data from {}'.format(zones_csv_path))\n",
    "    zones_dataset = zones_csv_path.as_df()\n",
    "    \n",
    "    target_path = path.join(context.artifact_path)\n",
    "    context.logger.info('Saving datasets to {} ...'.format(target_path))\n",
    "\n",
    "    # Store the data sets in your artifacts database\n",
    "    context.log_dataset('nyc-taxi-dataset', df=taxi_records_dataset, format='csv',\n",
    "                        index=False, artifact_path=target_path)\n",
    "    context.log_dataset('zones-dataset', df=zones_dataset, format='csv',\n",
    "                        index=False, artifact_path=target_path)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_zones_dict(zones_df):\n",
    "\n",
    "    # Remove unecessary fields\n",
    "    zones_df.drop(['Shape_Leng', 'Shape_Area', 'zone', 'LocationID', 'borough'], axis=1, inplace=True)\n",
    "    \n",
    "    # Convert DF to dictionary\n",
    "    zones_dict = zones_df.set_index('OBJECTID').to_dict('index')\n",
    "    \n",
    "    # Add lat/long to each zone\n",
    "    for zone in zones_dict:\n",
    "        shape = shapely.wkt.loads(zones_dict[zone]['the_geom'])\n",
    "        zones_dict[zone]['long'] = shape.centroid.x\n",
    "        zones_dict[zone]['lat'] = shape.centroid.y\n",
    "    \n",
    "    return zones_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_zone_lat(zones_dict, zone_id):\n",
    "    return zones_dict[zone_id]['lat']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_zone_long(zones_dict, zone_id):\n",
    "    return zones_dict[zone_id]['long']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "def transform_dataset(context : MLClientCtx, taxi_records_csv_path: DataItem, zones_csv_path: DataItem):\n",
    "    \n",
    "    context.logger.info('Begin datasets transform')\n",
    "    \n",
    "    context.logger.info('zones_csv_path: ' + str(zones_csv_path))\n",
    "    \n",
    "    zones_df = zones_csv_path.as_df()    \n",
    "    \n",
    "    # Get zones dictionary\n",
    "    zones_dict = get_zones_dict(zones_df)\n",
    "    \n",
    "    train_df = taxi_records_csv_path.as_df()\n",
    "    \n",
    "    # Clean DF\n",
    "    train_df = clean_df(train_df)\n",
    "    \n",
    "    # Enrich DF\n",
    "    train_df['pickup_latitude'] = train_df.apply(lambda x: get_zone_lat(zones_dict, x['PULocationID']), axis=1 )\n",
    "    train_df['pickup_longitude'] = train_df.apply(lambda x: get_zone_long(zones_dict, x['PULocationID']), axis=1 )\n",
    "    train_df['dropoff_latitude'] = train_df.apply(lambda x: get_zone_lat(zones_dict, x['DOLocationID']), axis=1 )\n",
    "    train_df['dropoff_longitude'] = train_df.apply(lambda x: get_zone_long(zones_dict, x['DOLocationID']), axis=1 )\n",
    "\n",
    "    train_df = add_datetime_info(train_df)\n",
    "    train_df = add_airport_dist(train_df)\n",
    "\n",
    "    train_df['pickup_latitude'] = radian_conv(train_df['pickup_latitude'])\n",
    "    train_df['pickup_longitude'] = radian_conv(train_df['pickup_longitude'])\n",
    "    train_df['dropoff_latitude'] = radian_conv(train_df['dropoff_latitude'])\n",
    "    train_df['dropoff_longitude'] = radian_conv(train_df['dropoff_longitude'])\n",
    "\n",
    "    train_df.drop(['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'congestion_surcharge', 'improvement_surcharge', 'pickup_datetime',\n",
    "                  'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'total_amount', 'RatecodeID', 'store_and_fwd_flag',\n",
    "                  'PULocationID', 'DOLocationID', 'payment_type'], \n",
    "                  axis=1, inplace=True, errors='ignore')\n",
    "    \n",
    "    # Save dataset to artifact\n",
    "    target_path = path.join(context.artifact_path)\n",
    "    context.log_dataset('nyc-taxi-dataset-transformed', df=train_df, artifact_path=target_path, format='csv')    \n",
    "    \n",
    "    context.logger.info('End dataset transform')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "params = {\n",
    "        'boosting_type':'gbdt',\n",
    "        'objective': 'regression',\n",
    "        'nthread': 4,\n",
    "        'num_leaves': 31,\n",
    "        'learning_rate': 0.05,\n",
    "        'max_depth': -1,\n",
    "        'subsample': 0.8,\n",
    "        'bagging_fraction' : 1,\n",
    "        'max_bin' : 5000 ,\n",
    "        'bagging_freq': 20,\n",
    "        'colsample_bytree': 0.6,\n",
    "        'metric': 'rmse',\n",
    "        'min_split_gain': 0.5,\n",
    "        'min_child_weight': 1,\n",
    "        'min_child_samples': 10,\n",
    "        'scale_pos_weight':1,\n",
    "        'zero_as_missing': True,\n",
    "        'seed':0,\n",
    "        'num_rounds':50000\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_model(context: MLClientCtx, input_ds: DataItem):\n",
    "    \n",
    "    context.logger.info('Begin training')\n",
    "    context.logger.info('LGBM version is ' + str(lgbm.__version__))\n",
    "    \n",
    "    train_df = input_ds.as_df()\n",
    "    \n",
    "    y = train_df['fare_amount']\n",
    "  \n",
    "    train_df = train_df.drop(columns=['fare_amount'])\n",
    "    train_df = train_df.drop(train_df.columns[[0]], axis=1)\n",
    "    x_train,x_test,y_train,y_test = train_test_split(train_df,y,random_state=123,test_size=0.10)\n",
    "    \n",
    "    train_set = lgbm.Dataset(x_train, y_train, silent=False,categorical_feature=['year','month','day','weekday'])\n",
    "    valid_set = lgbm.Dataset(x_test, y_test, silent=False,categorical_feature=['year','month','day','weekday'])\n",
    "    model = lgbm.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=500,verbose_eval=500, valid_sets=valid_set)\n",
    "    \n",
    "    context.log_model('FareModel',\n",
    "                     body=dumps(model),\n",
    "                     artifact_path='s3://mlrun-v1-warroom/',\n",
    "                     model_file=\"FareModel.pkl\")\n",
    "    \n",
    "    context.logger.info('End training')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
