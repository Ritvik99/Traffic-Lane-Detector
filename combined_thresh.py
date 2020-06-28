import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def absoluteH_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
	
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
    
	if orient == 'x':
		absoluteH_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	
    # Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*absoluteH_sobel/np.max(absoluteH_sobel))
	
	binary_output = np.zeros_like(scaled_sobel)
	
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	return binary_output

def magnitude_thresh(img, sobel_kernel=3, magnitude_thresh=(30, 100)):
	
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= magnitude_thresh[0]) & (gradmag <= magnitude_thresh[1])] = 1

	# Return the binary image
	return binary_output


def direction_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output


def hls_thresh(img, thresh=(100, 255)):
	
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	return binary_output


def combined_thresh(img):
	absoluteH_binary = absoluteH_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=255)
	magnitude_binary = magnitude_thresh(img, sobel_kernel=3, magnitude_thresh=(50, 255))
	direction_binary = direction_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
	hls_binary = hls_thresh(img, thresh=(170, 255))

	combined = np.zeros_like(direction_binary) #returns a null array of shape = direction_binary
	combined[(absoluteH_binary == 1 | ((magnitude_binary == 1) & (direction_binary == 1))) | hls_binary == 1] = 1

	return combined, absoluteH_binary, magnitude_binary, direction_binary, hls_binary 
