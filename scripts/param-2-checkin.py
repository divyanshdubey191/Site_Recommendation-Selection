# cd /Users/iprakhar22/Documents/MinorSpatial/scripts
# python param-2-checkin.py "Gastropub"

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

def normalize_function(arr, mx, mn):
    temp = []
    for item in arr:
        item = (item - mn) / (mx - mn)
        temp.append(item)

    return temp

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
    fig.canvas.set_window_title('Parameter - 2 - Checkin Count')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.4)
    ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    ax.axis('tight')
    plt.show()

def normalize_list(arr):
    mx = int(max(arr)[0])
    mn = int(min(arr)[0])

    for item in arr:
        item[0] = (item[0] - mn) / (mx - mn)

    return arr

def normalize_coordinates(coordinates, x_max, x_min, y_max, y_min):

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

def plot_clusters():
    color_map = ['r', 'b', 'g', 'c', 'y', 'm', 'k', 'gray', 'purple']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(normalized_coords.shape[0]):
        color_idx = clustering.labels_[i] % len(color_map)
        if clustering.labels_[i] != -1:
            ax.scatter(normalized_coords[i][0], normalized_coords[i][1], alpha = 0.8, c = color_map[color_idx])
    plt.show()


if __name__ == '__main__':

    category = sys.argv[1]

    l1=[]
    l2=[]
    d = pd.read_csv('../data/check_in_count_spread.csv')
    #Importing dataset
    df = pd.read_csv("../data/data-ny.csv", sep = ',', header=None, names =  ['userid', 'venueid', 'venuecatid', 'venuecatname','latitude','longitude','timezone','utctime'])
    df["check"]=1   
    #X=dataset.iloc[:,].values
    df = df.loc[df['venuecatname'] == category]

    df1 = df[['venueid','latitude','longitude','check']]
    # importing pandas package 

    df2=df1.groupby(['venueid','latitude','longitude']).sum()
    #print (df2[df2.check == 1])
    # df3=df2[df2.check == 1]
    # df4=df2[df2.check != 1]
    count=len(df2[df2['check'] == 1])
    for index, row in df2.iterrows():
        if(row['check']>100):
            row['check']=100
    per_one=count/(len(df2))


    for index, row in df2.iterrows():
        #print(index[0],index[1],index[2],row['check'])
        latitude=index[1]
        longitude=index[2]
        count=row['check']
        rad=0.005
        for i in range(0,count):
            coord_x=latitude
            coord_y=longitude
            choice1=random.randrange(0,2)
            change1=random.random()
            change1*=rad
            if(choice1==0):
                coord_x-=change1
                choice2=random.randrange(0,2)
                change2=random.random()
                change2*=rad
                if(choice2==0):
                    coord_y-=change2
                else:
                    coord_y+=change2
            else:
                coord_x+=change1
                choice2=random.randrange(0,2)
                change2=random.random()
                change2*=rad
                if(choice2==0):
                    coord_y-=change2
                else:
                    coord_y+=change2
            l1.append(coord_x)
            l2.append(coord_y)
        
    d['latitude']=l1
    d['longitude']=l2
    #d.to_csv('../data/check_in_count_spread.csv')

    coordinates = []
    for i in range(len(df['latitude'])):
        coordinates.append( [df['latitude'].tolist()[i] , df['longitude'].tolist()[i]] )

    #coordinates = [df['latitude'].tolist(), df['longitude'].tolist()]

    #print coordinates

    maxmin = pd.read_csv("../data/lat-lon-max-min.csv")

    normalized_coords = np.array(normalize_coordinates(coordinates, maxmin['latmax'], maxmin['latmin'], maxmin['lonmax'], maxmin['lonmin'] ))
    clustering = DBSCAN(eps=0.002, min_samples=2).fit(normalized_coords)

    #plot_clusters()

    height_data = get_height(clustering)
 
    #fitcurve(d['latitude'], d['longitude'], np.ones(len(d['longitude'])))
    fitcurve(height_data[... , 0], height_data[... , 1], height_data[... , 2])

