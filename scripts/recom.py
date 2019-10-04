#  python recom.py "Gastropub" 642 16
import pandas as pd
import math
import numpy as np
import sys, os
import webbrowser

#select all rows with category coffee shop
def get_rows_by_category(df, category):
	category_rows = df.loc[df['venuecatname'] == category]
	return category_rows


#get check incount of each venue with category coffee shop
def get_checkin_count_per_venue_in_category(category_rows):
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

#count number of venues per category
def get_number_of_venues_per_category(venue_group):
	count_venue = len(venue_group['venueid'])
	return count_venue


#count of how many times a user has gone to a particular venue
def get_sorted_user_checkin_count_at_venues(category_rows):
	user_venue_group = pd.DataFrame({'count' : category_rows.groupby(['userid','venueid']).size()}).reset_index()
	user_venue_group = user_venue_group.sort_values(['count'], ascending=False)
	return user_venue_group

def get_number_venues_visited_by_user(user_venue_group, user):
	user_rows = user_venue_group.loc[df['userid'] == user]
	count_user_venue = len(set(user_rows['venueid']))
	return count_user_venue


#list of all checkins made by a user
def get_all_user_checkins(df, user):
	user_rows = df.loc[df['userid'] == user]
	user_rows = pd.DataFrame({'count' : category_rows.groupby(['userid','venueid','latitude','longitude']).size()}).reset_index()
	return user_rows


#Calculate the Haversine distance between two geo co-ordiantes.
def haversine_dist(lat1, lon1, lat2, lon2):
	radius = 1000  # miles
	dlat = math.radians(lat2 - lat1)
	dlon = math.radians(lon2 - lon1)
	
	a = math.sin(dlat / 2) * math.sin(dlat / 2) + \
	math.cos(math.radians(lat1)) \
	* math.cos(math.radians(lat2)) * \
	math.sin(dlon / 2) * math.sin(dlon / 2)
	
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	d = radius * c
	return d


#number of checkins in venue radius / total number of checkins of each user
def calculate_p_close(venue_group, user_rows):
	x = 0
	d = []
	x = user_rows
	denom = sum(user_rows['count'])
	p_close = []

	print len(venue_group), len(user_rows)
	counter = 0
	for index, venue in venue_group.iterrows():
		user_rows = x
		counter += 1
		print counter
		for index1, user in user_rows.iterrows():
			d.append(haversine_dist(venue['latitude'], venue['longitude'], user['latitude'], user['longitude']))
		user_rows['distance'] = d
		maxi = max(d)
		mini = min(d)
		d = []
		user_rows = user_rows.loc[user_rows['distance'] < (maxi - mini)/3]
		p_close.append(sum(user_rows['count'])/denom)
	return p_close


#number of checkins at a venue / number of checkins at all venues belonging to the same category
def calculate_p_like(category_rows, venue_group, time):
	denom = sum(venue_group['count'])
	avg_time = pd.DataFrame({'average' : category_rows.groupby(['venueid'])['hour'].mean()}).reset_index()
	p_like = []
	#time penalty to make sure a place popular at nights is not suggested in the morning
	for index, venue in venue_group.iterrows():
		num = venue['count']
		initial_p_like = (num / denom)
		time_diff = abs(time - venue['avg_time'])
		if time_diff <= 3:
			penalized_p_like = initial_p_like
		elif time_diff <= 6:
			penalized_p_like = initial_p_like * 0.80
		elif time_diff <= 9:
			penalized_p_like = initial_p_like * 0.65
		elif time_diff <= 12:
			penalized_p_like = initial_p_like * 0.50
		else:
			penalized_p_like = initial_p_like * 0.30
		
		p_like.append(penalized_p_like)

	return p_like
	#distance between venue coordinates and each of user checkin, filter checkins using distance threshold and count #
"""
def calculate_p_checkin(p_like, df_venue):
	for i in range(len(p_like)):
		item = p_like[i]
		count = df_venue.loc[df_venue['venueid'] == item]['checkincount']
		p_like[i].append(count)

	# Binning on checkin-count

	mx_checkin = -1
	mn_checkin = int(1e7)

	for item in p_like:
		mx_checkin = max(mx_checkin,item[2])
		mn_checkin = min(mn_checkin,item[2])

	width = (mx_checkin-mn_checkin)/10

	for item in p_like:
		for i in range(10):
			low = mn_checkin + width*i
			high = low + 
	return
"""

def suggestions(p_checkin, p_close, venue_group, time):
	w1 = 0.6
	x = np.array(p_checkin)
	y = np.array(p_close)
	if(time <= 4): x=x[:8]; y=y[:8]
	elif(time <= 9): x=x[4:12]; y=y[4:12]
	elif(time <= 15): x=x[12:18]; y=y[12:18]
	p = w1 * x * y
	q = np.argsort(p)
	venueid = []
	end = 21 if len(q)>=21 else len(q)
	for i in range(1,end):
		index = q[i]
		venueid.append(venue_group.loc[index])
	print('\n')
	print("The suggested venueids for you are:")
	ret = []
	for venue in venueid:
		print(venue.venueid)
		ret.append(venue.venueid)
	return ret

def show_on_map(df_venue, suggested_ids):

	gmap_data = []
	count = 0
	
	for item in suggested_ids:
		row = df_venue.loc[df_venue['venueid'] == item]
		lat = float(row['latitude'])
		lng = float(row['longitude'])

		gmap_data.append([lat, lng, str(item)])

	pd.DataFrame(gmap_data).to_csv("../data/google-maps-data.csv", index = False)

	os.system("python map-flask-api.py")


if __name__ == '__main__':
	
	df = pd.read_csv("../data/data-ny.csv", sep = ',', header=None, names =  ['userid', 'venueid', 'venuecatid', 'venuecatname','latitude','longitude','timezone','utctime'])
	df_venue = pd.read_csv("../data/ny-checkin-count.csv", sep = ',', header=None, names =  ['venueid', 'latitude', 'longitude', 'checkincount'])
	df_venue = df_venue[1:]
	
	category = sys.argv[1]
	user = int(sys.argv[2])
	time = int(sys.argv[3])
	category_rows = get_rows_by_category(df, category)
	venue_group = get_checkin_count_per_venue_in_category(category_rows)
	count_venue = get_number_of_venues_per_category(venue_group)
	user_venue_group = get_sorted_user_checkin_count_at_venues(category_rows)
	count_user_venue = get_number_venues_visited_by_user(user_venue_group, user)
	user_rows = get_all_user_checkins(df, user)
	p_close = calculate_p_close(venue_group, user_rows)
	p_like = calculate_p_like(category_rows, venue_group, time)
	suggested_ids = suggestions(p_like, p_close, venue_group, time)

	#suggested_ids = category_rows['venueid']

	show_on_map(df_venue, suggested_ids)
