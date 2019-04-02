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
	bins=35
	pixels=img.shape[0]*img.shape[1]
	percentageup=pixels*0.3
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
	cv2.imwrite(folder+"_initial.tif",show)
elif os.path.isdir(folder):
	for fl in os.listdir(folder):
		img = cv2.imread(folder+"/"+fl)
		if img.rfind("_initial")>-1:
                        continue
                elif img.rfind("_Result")>-1:
                        continue
                elif img.rfind("_Seed")>-1:
                        continue
                elif img.rfind("_Result")>-1:
                        continue
                elif img.rfind("_Hessian")>-1:
                        continue
                elif img.rfind("_Morphsnakes")>-1:
                        continue
                elif img.rfind("_Segmentation")>-1:
                        continue
                elif img.rfind("_Noise")>-1:
                        continue
                elif img.rfind("_dilation")>-1:
                        continue
                elif img.rfind(".pck")>-1:
                        continue
		threshold=getThreshold(img)
		#show = img[:,:,2].copy()
		#show[show<threshold]=0
		#cv2.imwrite(folder+"/"+fl+"_initial.tif",show)
		#pl.imshow(show)
		#pl.show()
                
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
