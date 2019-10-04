# python site-selection-predict.py "Gastropub"

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os

wtparam1 = 10
wtparam2 = 4
wtparam3 = 5

def SI(arr):
	global wtparam1, wtparam2, wtparam3
	return (wtparam1 * arr[0] + wtparam2 * arr[1] + wtparam3 * arr[2]) / (wtparam1 + wtparam2 + wtparam3)

def normalize(lat, mx, mn):
	return (lat - mn)/(mx - mn)

def denormalize(lat, lng):
	denorm_lat = lat*(maxmin['latmax'] - maxmin['latmin']) + maxmin['latmin']
	denorm_lng = lng *(maxmin['lonmax'] - maxmin['lonmin']) + maxmin['lonmin']

	return denorm_lat, denorm_lng

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

def findmaxz():

	maxvalz = float(-1e18)
	corrlat = 0
	corrlng = 0

	for i in range(len(coordinates)):
		lat = coordinates[i][0]
		lng = coordinates[i][1]

		lat = float(normalize(lat, maxmin['latmax'], maxmin['latmin']))
		lng = float(normalize(lng, maxmin['lonmax'], maxmin['lonmin']))

		for j in range(0,len(X[0])-1):
			if float(XX[j])<=lat<=float(XX[j+1]):
				break

		closestlat = j

		for j in range(0,len(X[0])-1):
			if float(XX[j])<=lng<=float(XX[j+1]):
				break

		closestlng = j

		#print lat, lng, XX[closestlat], XX[closestlng]

		if Z[closestlat][closestlng] > maxvalz:
			maxvalz = Z[closestlat][closestlng]
			corrlat = closestlat
			corrlng = closestlng

	u, v = denormalize(XX[corrlat], YY[corrlng])
	return float(u), float(v)



if __name__ == '__main__':

	category = sys.argv[1]

	with open("../data/site-selection-coeff.csv", 'w') as fi:
		fi.write("a,b,c,d,e,f\n")

	print("Running Checkin plot")
	os.system("python param-2-checkin.py \"" + category + "\"")
	print("Running Population plot")
	os.system("python param-3-population.py \"" + category + "\"")
	print("Running Venue Density plot")
	os.system("python param-1-DBSCAN.py \"" + category + "\"")
	
	maxmin = pd.read_csv("../data/lat-lon-max-min.csv")

	df = pd.read_csv("../data/data-ny.csv", sep = ',', header=None, names =  ['userid', 'venueid', 'venuecatid', 'venuecatname','latitude','longitude','timezone','utctime'])
	df_venue = pd.read_csv("../data/ny-checkin-count.csv", sep = ',', header=None, names =  ['venueid', 'latitude', 'longitude', 'checkincount'])
	df_venue = df_venue[1:]

	

	category_rows = get_rows_by_category(df, category)
	coordinates = get_coordinates(df_venue, category_rows)


	maxmin = pd.read_csv("../data/lat-lon-max-min.csv")
	
	coeffs = pd.read_csv("../data/site-selection-coeff.csv")
	a = coeffs['a'].values.tolist()
	b = coeffs['b'].values.tolist()
	c = coeffs['c'].values.tolist()
	d = coeffs['d'].values.tolist()
	e = coeffs['e'].values.tolist()
	f = coeffs['f'].values.tolist()

	# Constants
	C = np.array([SI(a), SI(b), SI(c), SI(d), SI(e), SI(f)])
	#print C

	X,Y = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))

	XX = X.flatten()
	YY = Y.flatten()


	Z = C[4]*X**2 + C[5]*Y**2 + C[3]*X*Y + C[1]*X + C[2]*Y + C[0]

	# plot points and fitted surface
	fig = plt.figure()
	fig.canvas.set_window_title('Final Z function')
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.8)
	#ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
	plt.xlabel('X')
	plt.ylabel('Y')
	ax.set_zlabel('Z')
	ax.axis('equal')
	ax.axis('tight')
	plt.show()

	u,v = findmaxz()

	print ("Your site should be located near the coordinates :" , u , v)

	np.savetxt("../data/predicted-coords.csv", np.array([u,v]), delimiter=',')
	os.system("python final-predict-flask.py")
