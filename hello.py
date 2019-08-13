#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from itertools import product
from sklearn.preprocessing import StandardScaler
import time
import sys
import gc
import pickle
from math import ceil
from xgboost import XGBRegressor
from xgboost import plot_importance

#%%
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


#%%
train = pd.read_csv('C:/Users/Asus/Desktop/Data science/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('C:/Users/Asus/Desktop/Data science/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('C:/Users/Asus/Desktop/Data science/competitive-data-science-predict-future-sales/sample_submission.csv', error_bad_lines=False)
items = pd.read_csv('C:/Users/Asus/Desktop/Data science/competitive-data-science-predict-future-sales/items.csv', error_bad_lines=False)
shops = pd.read_csv('C:/Users/Asus/Desktop/Data science/competitive-data-science-predict-future-sales/shops.csv' ,error_bad_lines=False)
item_cats = pd.read_csv('C:/Users/Asus/Desktop/Data science/competitive-data-science-predict-future-sales/item_categories.csv' , error_bad_lines=False)


#%%
train.head()


#%%
test_shops = test.shop_id.unique()
train_expl = train[train.shop_id.isin(test_shops)]
test_items = test.item_id.unique()
train_expl = train_expl[train_expl.item_id.isin(test_items)]

#%%
MAX_BLOCK_NUM = train.date_block_num.max()
MAX_ITEM = len(test_items)
MAX_CAT = len(item_cats)
MAX_YEAR = 3
MAX_MONTH = 4 # 7 8 9 10
MAX_SHOP = len(test_shops)

#%%
grouped = pd.DataFrame(train_expl.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
num_graph = 10
id_per_graph = ceil(grouped.shop_id.max() / num_graph)
count = 0
for i in range(5):
    for j in range(2):
        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped[np.logical_and(count*id_per_graph <= grouped['shop_id'], grouped['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])
        count += 1


#%%
train_expl = train_expl.set_index('item_id').join(items.set_index('item_id')).drop('item_name', axis=1).reset_index()
train_expl.head()

#%%
train_expl['month'] = train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%m'))
train_expl['year'] = train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%Y'))
fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
num_graph = 10
id_per_graph = ceil(train.item_category_id.max() / num_graph)
count = 0
for i in range(5):
    for j in range(2):
        sns.pointplot(x='month', y='item_cnt_day', hue='item_category_id', 
                      data=train_expl[np.logical_and(count*id_per_graph <= train_expl['item_category_id'], train_expl['item_category_id'] < (count+1)*id_per_graph)], 
                      ax=axes[i][j])
        count += 1

#%%
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=train.item_cnt_day)

plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price)

#%%
    def eda(data):
        print("----------Top-5- Record----------")
        print(data.head(5))
        print("-----------Information-----------")
        print(data.info())
        print("-----------Data Types-----------")
        print(data.dtypes)
        print("----------Missing value-----------")
        print(data.isnull().sum())
        print("----------Null value-----------")
        print(data.isna().sum())
        print("----------Shape of Data----------")
        print(data.shape)
    
    def graph_insight(data):
        print(set(data.dtypes.tolist()))
        df_num = data.select_dtypes(include = ['float64', 'int64'])
        df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);
        
    def drop_duplicate(data, subset):
        print('Before drop shape:', data.shape)
        before = data.shape[0]
        data.drop_duplicates(subset,keep='first', inplace=True) #subset is list where you have to put all column for duplicate check
        data.reset_index(drop=True, inplace=True)
        print('After drop shape:', data.shape)
        after = data.shape[0]
        print('Total Duplicate:', before-after)

#%%
eda(train)
graph_insight(train)


#%%
eda(test)
graph_insight(test)

#%% 
eda(items)
graph_insight(items)

#%% 
eda(shops)

#%% 
eda(item_cats)


#%%
def unresanable_data(data):
        print("Min Value:",data.min())
        print("Max Value:",data.max())
        print("Average Value:",data.mean())
        print("Center Point of Data:",data.median())

#%% 
# -1 and 307980 looks like outliers, let's delete them
print('before train shape:', train.shape)
train = train[(train.item_price > 0) & (train.item_price < 300000)]
print('after train shape:', train.shape)


#%%
train.groupby('date_block_num').sum()['item_cnt_day'].hist(figsize = (20,4))
plt.title('Sales per month histogram')
plt.xlabel('Price')

plt.figure(figsize = (20,4))
sns.tsplot(train.groupby('date_block_num').sum()['item_cnt_day'])
plt.title('Sales per month')
plt.xlabel('Price')

#%%
unresanable_data(train['item_price'])
count_price = train.item_price.value_counts().sort_index(ascending=False)
plt.subplot(221)
count_price.hist(figsize=(20,6))
plt.xlabel('Item Price', fontsize=20);
plt.title('Original Distiribution')

plt.subplot(222)
train.item_price.map(np.log1p).hist(figsize=(20,6))
plt.xlabel('Item Price');
plt.title('log1p Transformation')
train.loc[:,'item_price'] = train.item_price.map(np.log1p)


#%%
count_price = train.date_block_num.value_counts().sort_index(ascending=False)
plt.subplot(221)
count_price.hist(figsize=(20,5))
plt.xlabel('Date Block');
plt.title('Original Distiribution')

count_price = train.shop_id.value_counts().sort_index(ascending=False)
plt.subplot(222)
count_price.hist(figsize=(20,5))
plt.xlabel('shop_id');
plt.title('Original Distiribution')

count_price = train.item_id.value_counts().sort_index(ascending=False)
plt.subplot(223)
count_price.hist(figsize=(20,5))
plt.xlabel('item_id');
plt.title('Original Distiribution')


#%% 
l = list(item_cats.item_category_name)
len(l)

#%%
l_cat = l

for ind in range(1,8):
    l_cat[ind] = 'Access'

for ind in range(10,18):
    l_cat[ind] = 'Consoles'

for ind in range(18,25):
    l_cat[ind] = 'Consoles Games'

for ind in range(26,28):
    l_cat[ind] = 'phone games'

for ind in range(28,32):
    l_cat[ind] = 'CD games'

for ind in range(32,37):
    l_cat[ind] = 'Card'

for ind in range(37,43):
    l_cat[ind] = 'Movie'

for ind in range(43,55):
    l_cat[ind] = 'Books'

for ind in range(55,61):
    l_cat[ind] = 'Music'

for ind in range(61,73):
    l_cat[ind] = 'Gifts'

for ind in range(73,77):
    l_cat[ind] = 'Soft'


item_cats['cats'] = l_cat
item_cats.head()

#%% 
train['date'] = pd.to_datetime(train.date,format="%d.%m.%Y")
train.head()

#%% 
## Pivot by monht to wide format
p_df = train.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0)
p_df.head()

#%%
## Join with categories
train_cleaned_df = p_df.reset_index()
train_cleaned_df['shop_id']= train_cleaned_df.shop_id.astype('str')
train_cleaned_df['item_id']= train_cleaned_df.item_id.astype('str')

item_to_cat_df = items.merge(item_cats[['item_category_id','cats']], how="inner", on="item_category_id")[['item_id','cats']]
item_to_cat_df[['item_id']] = item_to_cat_df.item_id.astype('str')

train_cleaned_df = train_cleaned_df.merge(item_to_cat_df, how="inner", on="item_id")

# Encode Categories
from sklearn import preprocessing

number = preprocessing.LabelEncoder()
train_cleaned_df[['cats']] = number.fit_transform(train_cleaned_df.cats)
train_cleaned_df = train_cleaned_df[['shop_id', 'item_id', 'cats'] + list(range(34))]
train_cleaned_df.head()

#%%
import xgboost as xgb
param = {'max_depth':10, 
         'subsample':1,
         'min_child_weight':0.5,
         'eta':0.3, 
         'num_round':1000, 
         'seed':1,
         'silent':0,
         'eval_metric':'rmse'}

progress = dict()
xgbtrain = xgb.DMatrix(train_cleaned_df.iloc[:,  (train_cleaned_df.columns != 33)].values, train_cleaned_df.iloc[:, train_cleaned_df.columns == 33].values)
watchlist  = [(xgbtrain,'train-rmse')]

bst = xgb.train(param, xgbtrain)
preds = bst.predict(xgb.DMatrix(train_cleaned_df.iloc[:,  (train_cleaned_df.columns != 33)].values))
from sklearn.metrics import mean_squared_error 
rmse = np.sqrt(mean_squared_error(preds,train_cleaned_df.iloc[:, train_cleaned_df.columns == 33].values))
print(rmse)
#%%
xgb.plot_importance(bst)
