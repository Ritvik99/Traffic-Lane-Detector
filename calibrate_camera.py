import numpy as np
import cv2
import pickle


def calibrate_camera():
	
	objPoint_dict = {
		1: (9, 5),
		2: (9, 6),
		3: (9, 6),
		4: (9, 6),
		5: (9, 6),
		6: (9, 6),
		7: (9, 6),
		8: (9, 6),
		9: (9, 6),
		10: (9, 6),
		11: (9, 6),
		12: (9, 6),
		13: (9, 6),
		14: (9, 6),
		15: (9, 6),
		16: (9, 6),
		17: (9, 6),
		18: (9, 6),
		19: (9, 6),
		20: (9, 6),
	}

	
	objPoint_list = []
	corners_list = []

	for k in objPoint_dict:
		nx, ny = objPoint_dict[k]

		objPoint = np.zeros((nx*ny,3), np.float32)
		objPoint[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) 

		fname = 'camera_cal/calibration%s.jpg' % str(k)
		img = cv2.imread(fname)

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

		if ret == True:
			
			objPoint_list.append(objPoint)
			corners_list.append(corners)

		else:
			print('return value = %s for %s from findChessboardCorners function' % (ret, fname))

	
	img = cv2.imread('camera_cal/straight_lines1.jpg')
	img_size = (img.shape[1], img.shape[0])
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoint_list, corners_list, img_size,None,None)

	return mtx, dist #camera matrix, distortion coefficients.


if __name__ == '__main__':
	mtx, dist = calibrate_camera()
	save_dict = {'mtx': mtx, 'dist': dist}
	with open('calibrate_camera.p', 'wb') as f:
		pickle.dump(save_dict, f)

