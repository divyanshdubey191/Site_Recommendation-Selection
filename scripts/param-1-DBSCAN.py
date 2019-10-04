# cd /Users/iprakhar22/Documents/MinorSpatial/scripts
# python param-1-DBSCAN.py "Gastropub"

from sklearn.cluster import DBSCAN
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import sys, os
import webbrowser
#from sympy import *

def save_coeff_to_file(C):
	f = open("../data/site-selection-coeff.csv", 'ab')
	for i in range(len(C)):
		C[i] = str(format(C[i], 'f'))
	np.savetxt(f, [C], delimiter=",")
	f.close()


def normalize_function(arr, mx, mn):
	temp = []
	for item in arr:
		item = (item - mn) / (mx - mn)
		temp.append(item)

	return temp

def fitcurve(x, y, z):
	data = np.c_[x,y,z]

	# regular grid covering the domain of the data
	#X,Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
	mn = np.min(data, axis=0)
	mx = np.max(data, axis=0)
	X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 60), np.linspace(mn[1], mx[1], 60))

	XX = X.flatten()
	YY = Y.flatten()

	# best-fit quadratic curve
	A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
	C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
	#print C
	save_coeff_to_file(C)
	
	#Z = C[4]*X**2 + C[5]*Y**2 + C[3]*X*Y + C[1]*X + C[2]*Y + C[0]
	Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

	# plot points and fitted surface
	fig = plt.figure()
	fig.canvas.set_window_title('Parameter - 1 - Venue Density')
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.4)
	ax.scatter(data[:,0], data[:,1], data[:,2], c='g', s=50)
	plt.xlabel('X')
	plt.ylabel('Y')
	ax.set_zlabel('Z')
	ax.axis('equal')
	ax.axis('tight')
	plt.show()

#select all rows with category specified
def get_rows_by_category(df, category):
	category_rows = df.loc[df['venuecatname'] == category]
	return category_rows

def get_coordinates(df_venue, rows):
	ids = rows['venueid']
	coords = []
	for item in ids:
		row = df_venue.loc[df_venue['venueid'] == item]
		lat = float(row['latitude'])
		lng = float(row['longitude'])

		coords.append([lat, lng])

	return coords

def plot_normalized_coords():
	plt.plot(normalized_coords[... , 0], normalized_coords[... , 1], 'ro')
	plt.show()

def plot_clusters():
	color_map = ['r', 'b', 'g', 'c', 'y', 'm', 'k', 'gray', 'purple']
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	for i in range(normalized_coords.shape[0]):
		color_idx = clustering.labels_[i] % len(color_map)
		if clustering.labels_[i] != -1:
			ax.scatter(normalized_coords[i][0], normalized_coords[i][1], alpha = 0.8, c = color_map[color_idx])
	plt.show()

# ---------------- TO FIX ----------------
def normalize_coordinates(coordinates, x_max, x_min, y_max, y_min):
	# y_max = x_max = -100000
	# y_min = x_min = -y_max

	# for item in coordinates:
	# 	y_max = max(y_max, item[1])
	# 	y_min = min(y_min, item[1])
	# 	x_max = max(x_max, item[0])
	# 	x_min = min(x_min, item[0])

	y_max = float(y_max)
	y_min = float(y_min)
	x_max = float(x_max)
	x_min = float(x_min)

	#print x_max, x_min, y_max, y_min

	for item in coordinates:
		item[0] = (item[0] - x_min) / (x_max - x_min)
		item[1] = (item[1] - y_min) / (y_max - y_min)
		#print item[0], item[1]

	return coordinates

def normalize_list(arr):
	mx = int(max(arr)[0])
	mn = int(min(arr)[0])

	for item in arr:
		item[0] = (item[0] - mn) / (mx - mn)

	return arr

def get_height(clustering):
	labels = clustering.labels_.reshape((normalized_coords.shape[0], 1))

	label_count = {}
	for i in labels:
		if int(i) in label_count :
			label_count[int(i)] += 1
		else:
			label_count[int(i)] = 0

	#print label_count
	for label in label_count :
		label_count[label] = min(label_count[label], 50)
	
	#print label_count

	height = np.zeros( labels.shape[0] )
	for i in range(labels.shape[0]):
		height[i] = label_count[ int(labels[i]) ]
	height = height.reshape((normalized_coords.shape[0], 1))

	height = np.array( normalize_list(height.tolist())).reshape((normalized_coords.shape[0], 1))

	# (x_coord, y_coord, height = cluster_size)
	height_data = np.hstack((normalized_coords, height))
	return height_data

if __name__ == '__main__':
	
	df = pd.read_csv("../data/data-ny.csv", sep = ',', header=None, names =  ['userid', 'venueid', 'venuecatid', 'venuecatname','latitude','longitude','timezone','utctime'])
	df_venue = pd.read_csv("../data/ny-checkin-count.csv", sep = ',', header=None, names =  ['venueid', 'latitude', 'longitude', 'checkincount'])
	df_venue = df_venue[1:]

	maxmin = pd.read_csv("../data/lat-lon-max-min.csv")

	category = sys.argv[1]

	category_rows = get_rows_by_category(df, category)
	coordinates = get_coordinates(df_venue, category_rows)
	normalized_coords = np.array(normalize_coordinates(coordinates, maxmin['latmax'], maxmin['latmin'], maxmin['lonmax'], maxmin['lonmin'] ))
	#print normalized_coords
	# plotting normalized points
	#plot_normalized_coords()

	clustering = DBSCAN(eps=0.002, min_samples=2).fit(normalized_coords)

	# plotting clusters
	#plot_clusters()
	
	height_data = get_height(clustering)

	#print height_data
	fitcurve(height_data[... , 0], height_data[... , 1], height_data[... , 2])
