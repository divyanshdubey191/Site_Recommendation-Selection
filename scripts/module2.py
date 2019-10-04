import pandas as pd
import math
from scipy.spatial import ConvexHull
import numpy as np
import surprise
import sys

def get_list_of_users(df):
	users = pd.DataFrame({'count' : df.groupby(['userid']).size()}).reset_index()
	#print(type(users))
	return users

def get_checkin_count_per_venue_in_category(category_rows):
	#get check incount of each venue with category coffee shop
	temp = category_rows
	time = pd.to_datetime(temp['utctime'])
	hour = []
	for i in time:
		hour.append(i.hour)
	temp.is_copy = False
	temp['hour'] = hour
	category_rows.is_copy = False
	category_rows["count"] =""
	category_rows["avg_time"] =""
	venue_group = temp.groupby(['venueid','longitude', 'latitude'])['hour'].agg({'count':'count', 'avg_time': 'mean'}).reset_index()
	#venue_group = pd.DataFrame({'count' : temp.groupby(['venueid', 'latitude', 'longitude']).size()}).reset_index()
	venue_group = venue_group.sort_values(['count'], ascending=False)
	return venue_group

def get_rows_by_category(df, category):
	category_rows = df.loc[df['venuecatname'] == category]
	return category_rows

def get_user_rows(df, user):
	user_rows = df.loc[df['userid'] == user]
	return user_rows

def get_number_of_venues_per_category(venue_group):
	#count number of venues per category
	count_venue = len(venue_group['venueid'])
	return count_venue

def get_number_venues_visited_by_user(user_venue_group, user):
	user_rows = user_venue_group.loc[user_venue_group['userid'] == user]
	count_user_venue = len(set(user_rows['venueid']))
	return count_user_venue

def get_all_user_checkins(df, user):
	#list of all checkins made by a user
	user_rows = df.loc[df['userid'] == user]
	user_rows = pd.DataFrame({'count' : category_rows.groupby(['userid','venueid','latitude','longitude']).size()}).reset_index()
	return user_rows

def get_average_users_per_venue():
	user = pd.DataFrame({'count' : df.groupby(['userid']).size()}).reset_index()
	venue = pd.DataFrame({'count' : df.groupby(['venueid']).size()}).reset_index()
	avg = len(venue) / len(user)
	return avg

def get_sorted_user_checkin_count_at_venues(category_rows):
	#count of how many times a user has gone to a particular venue
	user_venue_group = pd.DataFrame({'count' : category_rows.groupby(['userid','venueid']).size()}).reset_index()
	user_venue_group = user_venue_group.sort_values(['count'], ascending=False)
	return user_venue_group

def get_users_visited_venue(user_venue_group):
	venue_user = dict()
	for row in user_venue_group.iterrows():
	    if row[1].venueid in venue_user:
	        s = set()
	        s = venue_user[row[1].venueid]
	        s.add(row[1].userid)
	        venue_user[row[1].venueid] = s
	    else:
	        s = set()
	        s.add(row[1].userid)
	        venue_user[row[1].venueid] = s

def center_of_mass(users, df):
    avg_lat = []
    avg_long = []
    _users = users
    for index, row in _users.iterrows():
        user_rows = df.loc[df['userid'] == row['userid']]
        avg_lat.append(pd.DataFrame.mean(user_rows['latitude']))
        avg_long.append(pd.DataFrame.mean(user_rows['longitude']))
    #print(avg_lat[:5])
    _users['latitude'] = avg_lat
    _users['longitude'] = avg_long
    return _users

def get_checkins_per_venue(df):
	venue_checkins = pd.DataFrame({'count' : df.groupby(['venueid', 'latitude', 'longitude']).size()}).reset_index()	
	return venue_checkins

def get_lat_long(venueid, venue_checkins):
    row = venue_checkins.loc[venue_checkins['venueid'] == venueid ]
    return row['latitude'], row['longitude']

def haversine_dist(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two geo co-ordiantes."""
    radius = 3959  # miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d

def get_distance_from_center_of_mass(_users, venuelat, venuelong):
	dist = []
	print len(_users)
	for row in range(0,len(_users)):
		dist.append(haversine_dist(_users['latitude'][row], _users['longitude'][row], venuelat, venuelong))
	return dist

def calculate_p_close(venueid, df, users):
	venue_checkins = get_checkins_per_venue(df)
	venuelat, venuelong = get_lat_long(venueid, venue_checkins)
	_users = center_of_mass(users, df)
	dist = get_distance_from_center_of_mass(_users, venuelat, venuelong)
	p_close = [1 / x ** 1.64 for x in dist]
	return p_close

def calculate_p_go(count_venue, user_venue_group, users):
	#calculalte the probability of each user going to the venue
	p_go = []
	for user in users['userid']:
		count_user_venue = get_number_venues_visited_by_user(user_venue_group, user)
		p_go.append(count_user_venue / count_venue)
	return p_go

def calculate_p_like():
	p_like = []
	return p_like

def suggestions(p_go, p_like, p_close, users):
	x = np.array(p_go)
	y = np.array(p_close)
	z = np.array(p_like)
	p = x * y * z
	q = np.argsort(p)
	userid = []
	end = 21 if len(q)>=21 else len(q)
	for i in range(1,end):
		index = q[i]
		userid.append(users.loc[index])
	print('\n')
	print("The suggested users for the venue are:")
	for user in userid:
		print(int(user.userid))

if __name__ == '__main__':
	df = pd.read_csv("../data/data-ny.csv", sep = ',', header=None, names =  ['userid', 'venueid', 'venuecatid', 'venuecatname','latitude','longitude','timezone','utctime'])
	#category = "Coffee Shop"
	#venueid = '4ab966c3f964a5203c7f20e3'
	#user = 642
	category = sys.argv[1]
	venueid = sys.argv[2]
	category_rows = get_rows_by_category(df, category)
	venue_group = get_checkin_count_per_venue_in_category(category_rows)
	count_venue = get_number_of_venues_per_category(venue_group)
	user_venue_group = get_sorted_user_checkin_count_at_venues(category_rows)
	users = get_list_of_users(df)
	#print(users['userid'])
	p_go = calculate_p_go(count_venue, user_venue_group, users)
	p_close = calculate_p_close(venueid,df,users) 
	p_like = p_close
	suggestions(p_go, p_like, p_close, users)
