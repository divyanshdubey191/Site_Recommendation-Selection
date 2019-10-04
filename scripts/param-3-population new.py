# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalised(lat,ma,mi):
	for i in range(len(lat)):
		lat[i] = (float(lat[i])-min(lat)) / (max(lat)-min(lat))
		
	return lat


d2 = pd.read_csv("../data/lat-long-f.csv")
d3 = pd.read_csv("../data/ny-checkin-count.csv")
d = pd.read_csv("../data/lat-lon-max-min.csv")


lat2 = list(d2["latitude"])
lon2 = list(d2["longitude"])
lat3 = list(d3["Latitude"])
lon3 = list(d3["Longitude"])
pop = list(d2["2010"])
checkin = list(d3['Checkin Count'])

lat2_max = max(lat2)
lat2_min = min(lat2)
lon2_max = max(lon2)
lon2_min = min(lon2)
lat3_max = max(lat3)
lat3_min = min(lat3)
lon3_max = max(lon3)
lon3_min = min(lon3)

maxf = 1
minf = 0
maxlat = float(max(lat2_max,lat3_max))
minlat = float(min(lat2_min,lat3_min))
maxlon = float(max(lon2_max,lon3_max))
minlon = float(min(lon2_min,lon3_min))
m1 =[maxlat]
m2 = [minlat]
m3 =[maxlon]
m4 = [minlon]
d["latmax"] = m1
d["latmin"] = m2
d["lonmax"] = m3
d["lonmin"] = m4

d.to_csv("../data/lat-lon-max-min.csv")

	
for i in range(len(lat2)) :
	lat2[i] = (lat2[i]-minlat)/(maxlat-minlat)

for i in range(len(lat2)) :
	lon2[i] = (lon2[i]-minlon)/(maxlon-minlon)

for i in range(len(lat3)):
	lat3[i] = (lat3[i]-minlat)/(maxlat-minlat)

for i in range(len(lat3)):
	lon3[i] = (lon3[i]- minlon)/(maxlon-minlon)
	 
d2['Latitude']=lat2 
d2['Longitude']=lon2 
d3['Latitude']=lat3 
d3['Longitude']=lon3 

d2.drop(['latitude','longitude'],axis=1)
d3.drop(['Latitude','Longitude'],axis=1)


pop = normalised(pop,max(pop),min(pop)) 
d2.drop(['2010'],axis=1)
d2['2010'] = pop  

