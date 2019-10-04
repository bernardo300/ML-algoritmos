#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 01:19:56 2019

@author: bernardo
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
x , y = make_blobs(n_samples=200,centers=4)
kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
previsoes = kmeans.predict(x)
plt.scatter(x[:,0],x[:,1],c=previsoes)
previsoes = previsoes.reshape(-1,1)

centroides = kmeans.cluster_centers_
rotulos = kmeans.labels_
cores = ["green", "black", "blue","red"]
for i in range(x.shape[0]):
    plt.plot(x[i,0], x[i,1], previsoes[i])
    print("[{},{}] ->grupo {}".format(x[i][0], x[i][1], rotulos[i]))
plt.scatter(centroides[:,0], centroides[:,1], marker = "x")
