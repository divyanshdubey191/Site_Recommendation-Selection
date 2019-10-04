
import random
import pandas as pd
from sklearn.cluster import DBSCAN
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import numpy as np
import sys, os
import webbrowser

def save_coeff_to_file(C):
	f = open("../data/site-selection-coeff.csv", 'ab')
	for i in range(len(C)):
		C[i] = str(format(C[i], 'f'))
	np.savetxt(f, [C], delimiter=",")
	f.close()

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
	fig.canvas.set_window_title('Parameter - 3 - Locality Population')
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.4)
	ax.scatter(data[:,0], data[:,1], data[:,2], c='y', s=50)
	plt.xlabel('X')
	plt.ylabel('Y')
	ax.set_zlabel('Z')
	ax.axis('equal')
	ax.axis('tight')
	plt.show()


d1=pd.read_csv("../data/data-ny-2.csv")
d2=pd.read_csv("../data/lat-long-f.csv")
d3=pd.read_csv("../data/ny-checkin-count.csv")
d = pd.read_csv("../data/d.csv")

d1=d1.dropna(how='all')

lat1 = list(d1["latitude"])
lon1 = list(d1["longitude"])
lat2 = list(d2["latitude"])
lon2 = list(d2["longitude"])
lat3 = list(d3["Latitude"])
lon3 = list(d3["Longitude"])
pop = list(d2["2010"])
checkin = list(d3['Checkin Count'])
lat1_max = max(lat1)
lat1_min = min(lat1)
lon1_max = max(lon1)
lon1_min = min(lon1)
lat2_max = max(lat2)
lat2_min = min(lat2)
lon2_max = max(lon2)
lon2_min = min(lon2)
lat3_max = max(lat3)
lat3_min = min(lat3)
lon3_max = max(lon3)
lon3_min = min(lon3)


latmax = [lat1_max,lat2_max,lat3_max]
latmin = [lat1_min,lat2_min,lat3_min]
lonmax = [lon1_max,lon2_max,lon3_max]
lonmin = [lon1_min,lon2_min,lon3_min]

maxf = 1
minf = 0
maxlat = max(latmax)
minlat = min(latmin)
maxlon = max(lonmax)
minlon = min(lonmin)
m1 =[maxlat]
m2 = [minlat]
m3 =[maxlon]
m4 = [minlon]
d["latmax"] = m1
d["latmin"] = m2
d["lonmax"] = m3
d["lonmin"] = m4
d.to_csv("../data/lat-lon-max-min.csv")
def normalised(lat,ma,mi):
	for i in range(0,len(lat)):
		lat[i] = (((lat[i]-min(lat))/(max(lat)-min(lat)))*(maxf-minf)) + minf
		
	return lat

for i in range(0,len(lat1)):
	lat1[i] = (((lat1[i]-minlat)/(maxlat-minlat))*(maxf-minf)) + minf
	
for i in range(0,len(lat1)):
	lon1[i] = (((lon1[i]-minlon)/(maxlon-minlon))*(maxf-minf)) + minf    
	
for i in range(0,len(lat2)):
	lat2[i] = (((lat2[i]-minlat)/(maxlat-minlat))*(maxf-minf)) + minf
	
for i in range(0,len(lat2)):
	lon2[i] = (((lon2[i]-minlon)/(maxlon-minlon))*(maxf-minf)) + minf    

for i in range(0,len(lat3)):
	lat3[i] = (((lat3[i]-minlat)/(maxlat-minlat))*(maxf-minf)) + minf    
	
for i in range(0,len(lat3)):
	lon3[i] = (((lon3[i]-minlon)/(maxlon-minlon))*(maxf-minf)) + minf  
	
d1['Latitude']=lat1 
d1['Longitude']=lon1 
d2['Latitude']=lat2 
d2['Longitude']=lon2 
d3['Latitude']=lat3 
d3['Longitude']=lon3 

d1.drop(['latitude','longitude'],axis=1)
d2.drop(['latitude','longitude'],axis=1)
d3.drop(['Latitude','Longitude'],axis=1)


pop = normalised(pop,max(pop),min(pop)) 
d2.drop(['2010'],axis=1)
d2['2010'] = pop  

fitcurve(d2['Latitude'],d2['Longitude'],d2['2010'])
