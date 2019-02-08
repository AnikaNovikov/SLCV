
"""
SLCV
===========

These are the main function codes for the paper

  Rettig, A., Haase, T., Pletnyov, A., Kohl, B., Ertel, W., von Kleist, M., Sunkara, V., 
  "SLCV–A Supervised Learning - Computer Vision combined strategy for automated muscle 
  fibre detection in cross sectional images". Submitted to PeerJ in February 2019

"""



import numpy as np
import matplotlib.pyplot as pl
import cv2
import morphsnakes



def watershed(contours,ilastikpic):
	"""This function applies a distance transformation to each initial cluster, thresholds the transformation and applies the watershed algorithm.
	   In each initial cluster, multiple clusters may be found
	
	Parameters
	----------
	contours : list of cv2 contours (each contour is a list of boundaray points of this contour)
		All contours/initial clusters which exist in a segmented image (ilastik segmentation)
	ilastikpic : 2-dimensional array of type np.uint8
		pixel values 255 where a border was predicted by ilastik
	
	Returns
	-------
	list of cv2 contours
		All contours/final clusters which were generated from the initial contours by the Watershed algorithm
	"""
	
	threshold_parameter=0.3
	
	kernel = np.ones((3,3),np.uint8)
	seeds_list = []
	
	#for each individual cluster
	for i in range(len(contours)):
		cont_pic=draw_contour(contours[i],ilastikpic.shape)
		opening = cv2.morphologyEx(cont_pic,cv2.MORPH_OPEN,kernel,iterations=2)
		dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
		ret, foreground = cv2.threshold(dist_transform,threshold_parameter*dist_transform.max(),255,0)
		background = np.uint8(cv2.dilate(opening,kernel,iterations=3))
		ret, markers = cv2.connectedComponents(foreground)
		unknown = cv2.subtract(background,foreground)
		markers+=1
		markers[unknown==255]=0
		markers=cv2.watershed(ilastikpic,markers)
		label_num_cells = np.max(np.unique(markers))
		
		#each value in marker >= 2 represents a separated cluster piece
		#for each cell that the cluster has been separated into
		for i in range(2,(label_num_cells+1)):
			emp_img = np.zeros(markers.shape,dtype=np.uint8)
			emp_img[markers==i]=255
			im2,Conts,hierarchy = cv2.findContours(emp_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			if len(Conts)>0:
				seeds_list.append(Conts[0])
	
        return seeds_list




def GACSnake(snake_instance, seed)
	"""GACSnake uses the Snake implementation of Marquéz-Neila et al. (https://github.com/pmneila/morphsnakes#id1)
	   and applies it to each input seed
	
	Parameters
	----------
	snake_instance : an existing GAC snake object
		this object contains all necessary parameters
	seed : 2-dimensional array of type np.uint8
		binary image with the same dimensions as the input image has
	
	Returns
	-------
	2-dimensional array of type np.uint8
		reconstructed cell
	"""
	snake_instance.levelset = seed
	sett,stop = morphsnakes.evolve_visual(mgac, num_iters=200, background=None, visual=False)
	cell = (sett.astype(np.uint8))*255
	return cell




def SLCV_main(thresholdpic,ilastikpic):
	"""main SLCV function
	
	Parameters
	----------
	thresholdpic : 2-dimensional array of type np.uint8
		thresholded raw image 
		gaps are set to 0
	ilastikpic : 2-dimensional array of type np.uint8
		segmented image (ilastik segmentation)
		predicted border pixels are of value 255
	
	Returns
	-------
	list of cv2 contours
		final reconstructed cells
	"""
	
	#find contours, filter them, apply watershed
	keep_c = findContours(ilastikpic):
	keep_c = watershed(keep_c,ilastikpic)
	
	final_Contours = []
	
	#create snake instance
	floatIMG = thresholdpic/255.0
	gI = morphsnakes.gborders(floatIMG, alpha=2000, sigma=2)
	mgac = morphsnakes.MorphGAC(gI, smoothing=3, threshold=0.3, balloon=1)
	
        for c in keep_c:
		seed = draw_contour(c,ilastikpic.shape,binary=True)
		bgcheck = thresholdpic[seed==1]
                #if less than 30% of a seed are in a gap region, reconstruct it via snake
                if (np.sum(bgcheck==0)/float(len(bgcheck)) < 0.3):
			cell = GACSnake(mgac,seed)
			var = np.where(cell>0)
			#remove cells touching the image border
			if np.sum(var[0]==0) + np.sum(var[1]==0) + np.sum(var[0]==(ilastikpic.shape[0]-1)) + np.sum(var[1]==(ilastikpic.shape[1]-1)) == 0:
				im2,Con,hierarchy = cv2.findContours(cell,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				final_Contours.append(Con[0])





#helper functions
def draw_contour(contour,img_shape, binary=False):
	"""helper function which draws a contour object onto an empty image
	
	Parameters
	----------
	contours : cv2 contour object (list of boundary points)
		contour which is to be drawn
	img_shape : tuple of integers
		dimensions for the empty image
	binary : bool
		True if the image should be of type np.float, values between 0 and 1
		False if the image should be of type np.uint8, values between 0 and 255
	
	Returns
	-------
	2-dimensional array
		image with the contour of pixel value 1 or 255
	"""


def findContours(ilastik_cells,cont_find_it=2):
	"""helper function which finds contours in a segmented image. These contours are then filtered:
	   small contours consisting of border pixels are noise and are deleted
	   big contours containing smaller non-noise contours are delted
	
	Parameters
	----------
	ilastik_cells : 2-dimensional array of type np.uint8
		segmented image (ilastik segmentation)
	cont_find_it : int (default 2)
		number for the dilations applied to each initial cluster
	
	Returns
	-------
	list of cv2 contours
		filtered contours/initial clusters found in the segmented image
	"""
	
