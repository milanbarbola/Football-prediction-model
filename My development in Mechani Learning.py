#!/usr/bin/env python
# coding: utf-8

# # My own football prediction model 

# In this docummentation I will build a football prediction model from scratch. I will start by collecting some football data and after that I have it I will start building my model and I will approach the problem with different machine learning algorithm like randomforest and neuralnetwork. By the end of this documentation I hope that my model accuracy will be close enough or higher than 50%.

# ### Collecting the data

# In[2]:


import requests
import json
import pandas as pd
import concurrent.futures
import numpy as np
from datetime import date
import time
pd.set_option('display.max_columns', None)


# In[3]:


train_date_start = "2021/1/1"
train_date_end = "2022-6-1"
test_date_start = "2022/6/2"
test_date_end = "2022-12-31"
train_date = pd.date_range(train_date_start,train_date_end, freq ='d').strftime("%Y-%m-%d").tolist()
test_date = pd.date_range(test_date_start,test_date_end, freq = 'd').strftime("%Y-%m-%d").tolist()


# In[5]:


# Here I scrape urls which are gives me information about the name of the football matches and codes which I need to get more detailed statistics about the matches.
datas = [] # I will save the different days jason data in this list.
def getting_football_data(time):
  
    headers = {
      'authority': 'api.sofascore.com',
      'accept': '*/*',
      'accept-language': 'hu,en;q=0.9,hu-HU;q=0.8,en-US;q=0.7',
      'cache-control': 'max-age=0',
      'if-none-match': 'W/"25d995d50e"',
      'origin': 'https://www.sofascore.com',
      'referer': 'https://www.sofascore.com/',
      'sec-ch-ua': '"Not?A_Brand";v="8", "Chromium";v="108", "Google Chrome";v="108"',
      'sec-ch-ua-mobile': '?0',
      'sec-ch-ua-platform': '"Windows"',
      'sec-fetch-dest': 'empty',
      'sec-fetch-mode': 'cors',
      'sec-fetch-site': 'same-site',
      'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    }

    response = requests.get(f'https://api.sofascore.com/api/v1/sport/football/scheduled-events/{time}', headers=headers)
    data = json.loads(response.text)
    datas.append(data)


# In[6]:


# I use multithreading to accelerate the web scraping process
number_of_threads = len(test_date)
with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_threads) as executor:
  for time in test_date:
    executor.submit(getting_football_data,time)
print(len(datas))


# In[7]:


# here I process the data that I gained from the urls
home_name = []
away_name = []
number_of_goals_h= []
number_of_goals_a = []
number_of_goals_h_halftime= []
number_of_goals_a_halftime = []
winning_team = []
code_for_statistics = []
for data in datas:
  for team in data["events"]:
    try:
      if len(team["homeScore"]) > 4 and len(team["awayScore"]) > 4:
        home_name.append(team['homeTeam']["name"])
        away_name.append(team["awayTeam"]["name"])
        number_of_goals_h.append(team["homeScore"]["display"])
        number_of_goals_a.append(team["awayScore"]["display"])
        number_of_goals_h_halftime.append(team["homeScore"]["period1"])
        number_of_goals_a_halftime.append(team["awayScore"]["period1"])
        code_for_statistics.append(team["id"])
    except:
      continue
for home_goal,away_goal,count in zip(number_of_goals_h,number_of_goals_a,range(len(number_of_goals_a))):
  if int(home_goal) > int(away_goal):
    winning_team.append(1)
  if int(home_goal) == int(away_goal):
    winning_team.append(2)
  if int(home_goal) < int(away_goal):
    winning_team.append(3)
# 1 means that the home team won 2 that it was draw and 3 is that away team won


# In[8]:


# I arrange them into a DataFrame so when we reach the maching learning part we can work with it easily.
football_statistics = pd.DataFrame(
    [home_name,number_of_goals_h_halftime,away_name,number_of_goals_a_halftime,winning_team,code_for_statistics],
    columns=None, index=['Home', "Number_of_goals_home_halftime","Away","Number_of_goals_away_halftime","winner_team","stat_codes"]).T
football_statistics = football_statistics.replace(to_replace='', value=np.nan).dropna()


# In[10]:


#here I gather more statistical information about the football matches
all_statistics_data = []
def getting_statistics(id):
    url = f"https://api.sofascore.com/api/v1/event/{id}/statistics"
    headers = {
      'authority': 'api.sofascore.com',
      'accept': '*/*',
      'accept-language': 'hu,en;q=0.9,hu-HU;q=0.8,en-US;q=0.7',
      'cache-control': 'max-age=0',
      'origin': 'https://www.sofascore.com',
      'referer': 'https://www.sofascore.com/',
      'sec-ch-ua': '"Not?A_Brand";v="8", "Chromium";v="108", "Google Chrome";v="108"',
      'sec-ch-ua-mobile': '?0',
      'sec-ch-ua-platform': '"Windows"',
      'sec-fetch-dest': 'empty',
      'sec-fetch-mode': 'cors',
      'sec-fetch-site': 'same-site',
      'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    }

    response = requests.get(url, headers=headers)
    good_stat = []
    data = json.loads(response.text)
    for good in data:
        if good == "statistics":
            try:
                if len(data["statistics"]) == 3:
                  good_stat.append(id)
                  for period in data["statistics"]:
                    if period['period'] == "1ST":
                      good_stat.append(period)
            except:
                pass
            all_statistics_data.append(good_stat)


# In[11]:


# Using multithreading again because other wise we would never finish with the scraping within reasonable time
number_of_threads = len(code_for_statistics)
with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_threads) as executor:
    for cod in code_for_statistics:
        executor.submit(getting_statistics,cod)


# In[13]:


# in our list there are some statistic list which is empty even that we tried to scrape only urls whhich are full with statistics
for a in all_statistics_data:
    if len(a) == 0:
        all_statistics_data.remove(a)
print(len(all_statistics_data))


# In[14]:


#here I collect the ids of those matches which have detailed statistical information
good_ids = []
for a in all_statistics_data:
    try:
        good_ids.append(a[0])
    except:
        continue


# In[ ]:


# I dont realy remember what exactly I am doing here 
indexes = []
for ids in good_ids:
    try:
        indexes.append(football_statistics.loc[football_statistics.stat_codes == ids].index.tolist()[0])
    except:
        continue


# In[ ]:


football_statistics = football_statistics.loc[indexes].reset_index()
football_statistics.drop(["index"],axis = 1,inplace = True )
stat_codes = football_statistics.stat_codes.tolist()


# In[15]:


# inserting cells
football_statistics.insert(2, "Home_team_ball_possession",'')
football_statistics.insert(3,'Home_team_passes','')
football_statistics.insert(4,'Home_team_accurate_passes','')
football_statistics.insert(5,"Home_team_total_shots",'')
football_statistics.insert(6,'Home_team_shotson_target','')
football_statistics.insert(7,'Home_team_shotsoff_target','')
football_statistics.insert(8,'Home_team_blocked_shots','')
football_statistics.insert(9,'Home_team_corner_kicks','')
football_statistics.insert(10,'Home_team_yellow_cards','')

football_statistics.insert(13,"Away_team_ball_possession",'')
football_statistics.insert(14,'Away_team_passes','')
football_statistics.insert(15,'Away_team_accurate_passes','')
football_statistics.insert(16,"Away_team_total_shots",'')
football_statistics.insert(17,'Away_team_shotson_target','')
football_statistics.insert(18,'Away_team_shotsoff_target','')
football_statistics.insert(19,'Away_team_blocked_shots','')
football_statistics.insert(20,'Away_team_corner_kicks','')
football_statistics.insert(21,'Away_team_yellow_cards','')


# In[21]:


#processing the json files
for statistic in all_statistics_data:
    try:
        stat = str(statistic[1]).replace("\'", "\"")
        data = json.loads(stat)
        index = stat_codes.index(statistic[0])
        football_statistics.at[index,'Home_team_ball_possession'] = data['groups'][0]['statisticsItems'][0]['home']
        football_statistics.at[index,'Away_team_ball_possession'] = data['groups'][0]['statisticsItems'][0]['away']
        football_statistics.at[index,'Home_team_passes'] = data['groups'][4]['statisticsItems'][0]['home']
        football_statistics.at[index,'Away_team_passes'] = data['groups'][4]['statisticsItems'][0]['away']
        football_statistics.at[index,'Home_team_accurate_passes'] = data['groups'][4]['statisticsItems'][1]['home'].split('(')[0]
        football_statistics.at[index,'Away_team_accurate_passes'] = data['groups'][4]['statisticsItems'][1]['away'].split('(')[0]
        football_statistics.at[index,'Home_team_shotson_target'] = data['groups'][1]['statisticsItems'][1]['home']
        football_statistics.at[index,'Away_team_shotson_target'] = data['groups'][1]['statisticsItems'][1]['away']
        football_statistics.at[index,'Home_team_total_shots'] = data['groups'][1]['statisticsItems'][0]['home']
        football_statistics.at[index,'Away_team_total_shots'] = data['groups'][1]['statisticsItems'][0]['away']
        football_statistics.at[index,'Home_team_shotsoff_target'] = data['groups'][1]['statisticsItems'][2]['home']
        football_statistics.at[index,'Away_team_shotsoff_target'] = data['groups'][1]['statisticsItems'][2]['away']
        football_statistics.at[index,'Home_team_blocked_shots'] = data['groups'][1]['statisticsItems'][3]['home']
        football_statistics.at[index,'Away_team_blocked_shots'] = data['groups'][1]['statisticsItems'][3]['away']
        football_statistics.at[index,'Home_team_corner_kicks'] = data['groups'][2]['statisticsItems'][0]['home']
        football_statistics.at[index,'Away_team_corner_kicks'] = data['groups'][2]['statisticsItems'][0]['away']
        football_statistics.at[index,'Home_team_yellow_cards'] = data['groups'][2]['statisticsItems'][2]['home']
        football_statistics.at[index,'Away_team_yellow_cards'] = data['groups'][2]['statisticsItems'][2]['away']

    
    except:
        continue


# In[22]:


# If there is still a row in which we have a missing value than I delete this row because It is more than sure than all the value will missing from this row
football_statistics= football_statistics.replace(to_replace='None', value=np.nan).dropna()


# In[24]:


# here you can save out your dataframe but you have to unhasteg it first
#football_statistics.to_csv('fotball_statistics_full.csv', index=False)
#football_statistics.to_csv('football_statistics_test.csv',index = False)


# ## Machine Learning algorithms

# ### Bringig my data into the right format

# In[282]:


# loading and spliting your data into train and validation
data = pd.read_csv("/v/du5usz/Intro to Machine Learning/fotball_statistics_full.csv")
test_data = pd.read_csv("/v/du5usz/Intro to Machine Learning/football_statistics_test.csv")


# In[283]:


data.head() # The first and most important thing before you start building your model is to analyze the data that you will
# work with


# In[284]:


# drop the unnecesary columns
data.drop(["Home"],axis = 1,inplace = True )
data.drop(["Away"],axis = 1,inplace = True )
data.drop(["stat_codes"],axis = 1,inplace = True )
# test data
test_data.drop(["Home"],axis = 1,inplace = True )
test_data.drop(["Away"],axis = 1,inplace = True )
test_data.drop(["stat_codes"],axis = 1,inplace = True )


# In[285]:


#there are some row in which the datas are incorrect so we remove these rows
index_home = []
home_list = data.Home_team_ball_possession.tolist()
for a in home_list:
    if '.' in a:
        index_home.append(home_list.index(a))
data = data.drop(index_home, axis=0)
# test data
index_home_test = []
home_list_test = test_data.Home_team_ball_possession.tolist()
for a in home_list_test:
    if '.' in a:
        index_home_test.append(home_list_test.index(a))
test_data = test_data.drop(index_home_test, axis=0)


# In[286]:


# we replace all the numbers which are in percentage form and we make an int number out of it
data.Home_team_ball_possession = data.Home_team_ball_possession.map(lambda p: int(p.replace("%","")))
data.Away_team_ball_possession = data.Away_team_ball_possession.map(lambda p: int(p.replace("%","")))
#test data
test_data.Home_team_ball_possession = test_data.Home_team_ball_possession.map(lambda p: int(p.replace("%","")))
test_data.Away_team_ball_possession = test_data.Away_team_ball_possession.map(lambda p: int(p.replace("%","")))


# In[287]:


data = data.dropna()
test_data = test_data.dropna()


# In[288]:


data.reset_index(inplace = True)
data.drop("index",axis = 1, inplace = True)

test_data.reset_index(inplace = True)
test_data.drop("index",axis = 1, inplace = True)


# In[289]:


# it is possible to convert a column of one type into another
# we want to transform all the columns from existing int64 data type into a float32 data type
data.astype('float32')
test_data.astype('float32')


# I only realised at this point that it is absolutely useless even it is make my model precision worse to keep the home stat and away stat nearby near in the dataframe so now I am going to correct it and reformet in a form that under the home stat there will be the away stat and moreover with this I duplicated the amount of statistics that I previously had.
# I made a realisation again that it doesn't even matter that the statistic is home or away the only thing that matters that what was the outcame based on the statistical data. So I will rename my column names by ball posession and name like this because it is irrelevant if it was a home team stat or an away team stat. Maybe in the future we could expand our dataframe with data like whether the team played on a home field or he was playing on an away field and many more details that can influence the outcome of the football match

# In[290]:


#spliting the dataframe at home and away part
home_column_names = data.columns[:10].tolist()
home_column_names.append('winner_team')
away_column_names = data.columns[10:20].tolist()
away_column_names.append('winner_team')
data1 = data[home_column_names]
data2 = data[away_column_names]
test_data1 = test_data[home_column_names]
test_data2 = test_data[away_column_names]


# In[291]:


#deleting the column names
data1.columns = [''] * len(data1.columns) 
data2.columns = [''] * len(data2.columns) 
test_data1.columns = [''] * len(test_data1.columns) 
test_data2.columns = [''] * len(test_data2.columns) 


# In[292]:


# give column names to the dataframe
data1.columns = ['Number_of_goals_halftime','ball_possessio','passes','accurate_passes','total_shots','shotson_target','shotsoff_target','blocked_shots','corner_kikcs','yellow_cards','winner_team']
data2.columns = ['Number_of_goals_halftime','ball_possessio','passes','accurate_passes','total_shots','shotson_target','shotsoff_target','blocked_shots','corner_kikcs','yellow_cards','winner_team']
test_data1.columns = ['Number_of_goals_halftime','ball_possessio','passes','accurate_passes','total_shots','shotson_target','shotsoff_target','blocked_shots','corner_kikcs','yellow_cards','winner_team']
test_data2.columns = ['Number_of_goals_halftime','ball_possessio','passes','accurate_passes','total_shots','shotson_target','shotsoff_target','blocked_shots','corner_kikcs','yellow_cards','winner_team']


# In[293]:


# appending the dataframes
data = pd.concat([data1,data2],ignore_index = True)
test_data = pd.concat([test_data1,test_data2],ignore_index = True)


# In[305]:


# now that we analyzed and reformed our data we can split our data into train and validation
y = data.winner_team
football_prediction = data.drop(['winner_team'],axis = 1)
X = football_prediction.select_dtypes(exclude = ['object']) # here we choose columns with only numerical data
X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size= 0.8,test_size = 0.2,random_state = 0)

y_test = test_data.winner_team
football_prediction_test = test_data.drop(['winner_team'],axis = 1)
X_test = football_prediction_test.select_dtypes(exclude = ['object'])


# ### RandomForestRegressor

# In[306]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# In[307]:


# define a function whit which you can measure the precision of your model
def precision(y_valid,preds):
    good_prediction = 0
    pred_list = preds.tolist()
    valid_list = y_valid.tolist()
    for valid,pred in zip(pred_list,valid_list):
        valid = float(round(valid))
        if valid == pred:
            good_prediction +=1
    percentage = (good_prediction/len(pred_list))*100
    return percentage


# In[308]:


# define function to measure quality of your model
def score_dataset(X_train, X_valid, y_train, y_valid,estimator,random,leaf):
    model = RandomForestRegressor(n_estimators = estimator, random_state = random,max_leaf_nodes = leaf)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return precision(y_valid,preds)


# In[309]:


prediction_list = []
leaf_list = [500,1000,1200,1250]
for random in range(1,10):
    for leaf in leaf_list:
        pred = []
        pred.append(score_dataset(X_train,X_valid,y_train,y_valid,1,random,leaf))
        pred.append(1)
        pred.append(random)
        pred.append(leaf)
        prediction_list.append(pred)
        #print(score_dataset(X_train,X_valid,y_train,y_valid,estimator,random,leaf))
#print("The best prediction was:",max(prediction_list))


# In[310]:


max_num = []
for a in prediction_list:
    max_num.append(a[0])
max_index = max_num.index(max(max_num))
print(prediction_list[max_index])


# In[311]:


# here I train my model with the best parameters
model = RandomForestRegressor(n_estimators = 1, random_state = prediction_list[max_index][2],
                              max_leaf_nodes = prediction_list[max_index][3])
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(precision(y_test,preds))


# ### RandomForestClassifier

# In[312]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[321]:


min_sample_split = [2,4,6,8,10,12,15,20]
leaf = [1,2,3,4,6,8,10]
random_state = [2,4,6,8,10]
n_jobs = [-1,2,4,-3]
estimator = [800,1000,1200,1500]
for y in estimator:
    for a in min_sample_split:
        for b in leaf:
            for c in random_state:
                for d in n_jobs:
                    random_forest = RandomForestClassifier(criterion='gini', 
                                 n_estimators=y,
                                 min_samples_split=a,
                                 min_samples_leaf=b,
                                 max_features='auto',
                                 oob_score=True,
                                 random_state=c,
                                 n_jobs=d)
                    random_forest.fit(X_train, y_train)
                    pred = random_forest.predict(X_test)
                    good = 0
                    for t,r in zip(pred.tolist(),y_test):
                        if t == r:
                            good += 1
                    if good/len(y_test) > 45:
                        print('The precision is:',good/len(y_test))
                        print(y,a,b,c,d)


# In[318]:


good = 0
for a,b in zip(pred.tolist(),y_test):
    if a == b:
        good += 1
print('The precision is:',good/len(y_test))


# In[ ]:




