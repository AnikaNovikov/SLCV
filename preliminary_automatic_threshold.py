from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as pl
import cv2
import os
import sys

lowerthr=50
upperthr=150

if len(sys.argv)<2:
	print("Please input a folder of raw images or the path to a single raw image")
else:
	folder=sys.argv[1]


def getThreshold(img):
        """This function automatically attempts to threshold an image to exclude gap regions
	
	Parameters
	----------
	img : 3-dimensional array of type np.uint8
		raw image
	
	Returns
	-------
	int
		The suggested threshold value for the image
	"""
	bins=35
	pixels=img.shape[0]*img.shape[1]
	#fix upper limit percentage of pixels expected to be a gap
	percentageup=pixels*0.3
	#fix plausible lower percentage of pixels expected to be a gap
	percentagedown=pixels*0.05
	show=0
	peaks=[[0]]
	choose=0
	iter=0
	#while there are not enough pixels assigned to a gap region and bin size is not too small
	while np.sum(show)<percentagedown and iter<5:
		a=np.histogram(img[:,:,2],bins=bins)
		#determine the local maxima of the intensity histogram
		peaks=argrelextrema(a[0],np.greater)
		#extract the first local maximum
		threshold=int(round(a[1][peaks[0][0]+1]))
		show=img[:,:,2]<threshold
		#increase bin size to make histogram more fine graded, make maxima easier detectable
		bins=bins+15
		iter+=1

	#still too few pixels assigned to a gap region, smaller bin size does not help
	if np.sum(show)<percentagedown and iter>=5:
		bins=35
		show=0
		peaks=[[0]]
		iter=0
		#repeat the process like before
		while np.sum(show)<percentagedown and iter<5:
			a=np.histogram(img[:,:,2],bins=bins)
			peaks=argrelextrema(a[0],np.greater)
			#extract the second maximum
			threshold=int(round(a[1][peaks[0][1]+1]))
			show=img[:,:,2]<threshold
			bins=bins+15
			iter+=1
        
	#if now too many pixels are assigned to be in a gap, accept the solution with the next bigger bin size
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
        
	#if none of the above strategies help, give up
	if bins<=15:
		print("failed")
		threshold=0
	
	return threshold

if os.path.isfile(folder):
	img = cv2.imread(folder)
	threshold=getThreshold(img)
	show = img[:,:,2].copy()
	show[show<threshold]=0
	cv2.imwrite(folder+"_initial.tif",show)
elif os.path.isdir(folder):
	for fl in os.listdir(folder):
		img = cv2.imread(folder+"/"+fl)
		threshold=getThreshold(img)
                
                new_IMG = img.copy()
                new_IMG[:,:,2] = np.where(img[:,:,2] < threshold, 0, img[:,:,2])
        	shift = (lowerthr-threshold)
                
                nonzero = np.where(new_IMG[:,:,2] == 0, False, True)
        	new_IMG_2 = new_IMG.copy().astype(np.uint16)
        	new_IMG_2[:,:,2] = np.where(nonzero, new_IMG_2[:,:,2] + shift, new_IMG_2[:,:,2])
        	new_IMG_2[:,:,2] = np.where(new_IMG_2[:,:,2] > 50000, 0, new_IMG_2[:,:,2])
                new_IMG_2[:,:,2] = np.where(new_IMG_2[:,:,2] > 255, 255, new_IMG_2[:,:,2])
                new_IMG_2 = new_IMG_2.astype(np.uint8)
                
                cv2.imwrite(folder+"/"+fl[:-4]+"_initial.tif",new_IMG_2)


		
else:
	print("Input is neither a file nor a directory, exiting...")
