from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as pl
import cv2
import os
import sys

if len(sys.argv)<2:
	print("Please input a folder of raw images or the path to a single raw image")
else:
	folder=sys.argv[1]

def getThreshold(img):
	bins=35
	pixels=img.shape[0]*img.shape[1]
	percentageup=pixels*0.4
	percentagedown=pixels*0.05
	show=0
	peaks=[[0]]
	choose=0
	iter=0
	while np.sum(show)<percentagedown and iter<5:
		a=np.histogram(img[:,:,2],bins=bins)
		peaks=argrelextrema(a[0],np.greater)
		threshold=int(round(a[1][peaks[0][0]+1]))
		show=img[:,:,2]<threshold
		bins=bins+15
		iter+=1
	
	if np.sum(show)<percentagedown and iter>=5:
		bins=35
		show=0
		peaks=[[0]]
		iter=0
		while np.sum(show)<percentagedown and iter<5:
			a=np.histogram(img[:,:,2],bins=bins)
			peaks=argrelextrema(a[0],np.greater)
			threshold=int(round(a[1][peaks[0][1]+1]))
			show=img[:,:,2]<threshold
			bins=bins+15
			iter+=1
	
	while np.sum(show)>percentageup and bins>15:
		bins=bins-15
		a=np.histogram(img[:,:,2],bins=bins)
		peaks=argrelextrema(a[0],np.greater)
		if len(peaks[0])==0:
			threshold=0
			print("failed")
			return threshold
		else:
			threshold=int(round(a[1][peaks[0][0]+1]))
		show=img[:,:,2]<threshold
	
	if bins<=15:
		print("failed")
		threshold=0
	
	return threshold

if os.path.isfile(folder):
	img = cv2.imread(folder)
	threshold=getThreshold(img)
	show = img[:,:,2].copy()
	show[show<threshold]=0
	cv2.imwrite(folder+"_thresholded.tif",show)
elif os.path.isdir(folder):
	for fl in os.listdir(folder):
		img = cv2.imread(folder+"/"+fl)
		threshold=getThreshold(img)
		show = img[:,:,2].copy()
		show[show<threshold]=0
		cv2.imwrite(folder+"/"+fl+"_thresholded.tif",show)
		#pl.imshow(show)
		#pl.show()
else:
	print("Input is neither a file nor a directory, exiting...")