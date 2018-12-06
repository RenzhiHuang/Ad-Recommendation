from data_import import *


path = './yahoo_ad_clicks.csv'
raw_data = data_import(path)
print(type(raw_data))
print(raw_data.shape)