import cv2
import os
import numpy as np
from Oxford_dataset import ReadCameraModel , UndistortImage 
import matplotlib.pyplot as plt

path = 'stereo/centre/'
list_of_filenames = [filename for filename in os.listdir(path)]
# print(list_of_filenames)
dataset_size = len(list_of_filenames)
print(dataset_size)
list = []
homogenous1 = np.identity(4)
t1 = np.array([[0, 0, 0, 1]])
t1 = t1.T

def Homogenousmatrix(R, t):
    h = np.column_stack((R, t))
    a = np.array([0, 0, 0, 1])
    h = np.vstack((h, a))
    return h
# Extract the camera parameters using ReadCameraModel function
fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('model/')
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
frame_count = 0
for im in range(19, dataset_size-1):
        frame_count = frame_count + 1

        img1 = cv2.imread(path + '/'+list_of_filenames[im],0)
        img2 = cv2.imread(path + '/' + list_of_filenames[im+1], 0)

        # Convert image from Bayer format to BGR format
        image1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
        image2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)

        # # Extract the camera parameters using ReadCameraModel function
        # fx, fy, cx, cy, G_camera_image, LUT = rd.ReadCameraModel('Oxford_dataset/model')

        # Undistort the current input image and next frame using UndistortImage function.
        undistorted_image1 = UndistortImage(image1, LUT)
        undistorted_image2 = UndistortImage(image2, LUT)


        # cv2.imshow("img",img)
        # cv2.imshow("image",image)
        # cv2.imshow("undistort",undistorted_image)

        img1 = undistorted_image1
        img2 = undistorted_image2

        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # BFMatcher with default params
        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(des1, des2, k=2)
        FLANN_INDEX_KDTREE = 0
        index = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index, search)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []

        # Apply ratio test
        for m, n in matches:
                if m.distance < 1 * n.distance:
                        good.append(m)

        # src_pts = np.int32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        src_pts = np.int32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.int32([kp2[m.trainIdx].pt for m in good])
        # dst_pts = np.int32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
        src_pts = src_pts[mask.ravel() == 1]
        dst_pts = dst_pts[mask.ravel() == 1]
        # matchesMask = mask.ravel().tolist()
        E = np.matmul(np.matmul(K.T, F), K)
        ret, R, T, mask = cv2.recoverPose(E, src_pts, dst_pts, K)
        print(R)
        homogenous2 = Homogenousmatrix(R, T)
        homogenous1 = np.matmul(homogenous1, homogenous2)
        p = np.matmul(homogenous1, t1)

        plt.scatter(p[0][0], -p[2][0], color='r')
        plt.savefig("Output3/output_" + str(frame_count) + '.png')


        if cv2.waitKey(1) == 27:
                break
plt.legend(['inbuilt'])
plt.show()
cv2.destroyAllWindows()
