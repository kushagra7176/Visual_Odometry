

from Oxford_dataset import ReadCameraModel as rd, UndistortImage as ud
import numpy as np
import random
from numpy.linalg import matrix_rank
import cv2
import matplotlib.pyplot as plt
import os

frames = []

'''
 Function to check and correct the camera pose.
'''
def CheckCameraPose(C, R):
    newC, newR = [], []

    for i in range(len(R)):
        if np.linalg.det(R[i]) < 0:
            newC.append(-C[i].reshape(3,1))
            newR.append(-R[i])
        else:
            newC.append(C[i].reshape(3,1))
            newR.append(R[i])

    return newC, newR

'''
 Function to Estimate the Camera Pose from the Essential Matrix.
'''
def cameraPose(E):
    U, D, V = np.linalg.svd(E, full_matrices=True)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    C1 = U[:, 2]
    C2 = -U[:, 2]
    C3 = U[:, 2]
    C4 = -U[:, 2]
    C = [C1, C2, C3, C4]

    R1 = np.matmul(np.matmul(U, W), V)
    R2 = np.matmul(np.matmul(U, W), V)
    R3 = np.matmul(np.matmul(U, W.T), V)
    R4 = np.matmul(np.matmul(U, W.T), V)
    R = [R1, R2, R3, R4]

    newC, newR = CheckCameraPose(C, R)
    newC = np.array(newC)
    newR = np.array(newR)

    return  newR, newC

'''
 Function to Estimate Essential Matrix from the Fundamental Matrix.
'''
def estimate_Essential_Matrix( K, F) -> np.array:
    E = K.T @ F @ K
    U, S, V = np.linalg.svd(E)
    S = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    E = U @ S @ V
    return E

'''
 Function to generate the Homogenous Matrix from a giver Rotation and Translation.
'''
def Homogenousmatrix(rot_mat, t):
    i = np.column_stack((rot_mat, t))
    a = np.array([0, 0, 0, 1])
    H = np.vstack((i, a))
    return H

'''
 Function that generates the 3D reconstructed points from the given two camera matrices
 and the the Key Points from two consecutive images.
'''
def generateReconstructedPoints( P_origin, P_new, KeyPts1, KeyPts2) :
    P1 = P_origin
    P2 = P_new
    X = []
    for i in range(len(KeyPts1)):
        x1 = KeyPts1[i]
        x2 = KeyPts2[i]

        A = np.array([x1[0] * P1[2,:] - P1[0,:] ,
                      x1[1] * P1[2,:] - P1[1,:],
                      x2[0] * P2[2,:] - P2[0,:] ,
                      x2[1] * P2[2,:] - P2[1,:]])

        U, S, V = np.linalg.svd(A)
        V = V[3]
        V = V / V[-1]

        X.append(V)
    return X

'''
 Function that performs Linear Triangulation and generates the final Rotation and Translation.
'''
def linear_triangulation(RotationMatrix, Tlist, K, inlier1, inlier2):
    flag = 0
    for p in range(4):
        # The Rotation and Translation at the origin of world frame.
        R_origin = np.eye(3, 3)
        C_origin = [[0], [0], [0]]

        # The Rotation and Translation of the camera wrt world frame.
        R_new = RotationMatrix[p]
        C_new = Tlist[p]

        # Compute the Camera matrix at origin and new frame using the camera intrinsic parameter
        P_origin = K @ np.hstack((R_origin, -R_origin @ C_origin))
        P_new = K @ np.hstack((R_new, -R_new @ C_new))

        # Generate all the reconstructed points from the key points.
        X = generateReconstructedPoints( P_origin,P_new, inlier1, inlier2)
        X = np.array(X)

        # Check the Cheirality condition for all the reconstructed points.
        count = 0
        for i in range(X.shape[0]):
            x = X[i, :].reshape(-1, 1)
            if R_new[2] @ np.subtract(x[0:3], C_new) > 0 and x[2] > 0: count += 1

        # Store the index of the Rotation and Translation that generates the maximum number of reconstructed points in from of the camera.
        # The number of reconstructed points in front of the camera is stored in the variable "count".
        if flag < count:
            flag, ind = count, p

    # Store the final Rotation and Translation values.
    finR = RotationMatrix[ind]
    finC = Tlist[ind]

    return finR, finC

'''
Function to compute the fundamental matrix
'''
def calc_fundamental_mat(Key_pts1,Key_pts2):
    A = np.empty((8, 9))
    # print("A", A)

    for i in range(len(Key_pts1) - 1):
        x1 = Key_pts1[i][0]
        y1 = Key_pts1[i][1]
        x2 = Key_pts2[i][0]
        y2 = Key_pts2[i][1]
        A[i] = np.array([x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])

    u, s, v = np.linalg.svd(A)
    f = v[-1].reshape(3, 3)
    u_1, s_1, v_1 = np.linalg.svd(f)
    s_2 = np.array([[s_1[0], 0, 0], [0, s_1[1], 0], [0, 0, 0]])
    F = np.dot(np.dot(u_1, s_2), v_1)

    return F

'''
 function to calculate error for RANSAC
'''
def CheckCondition(x1,x2,F):
    x11=np.array([x1[0],x1[1],1])
    x22=np.array([x2[0],x2[1],1]).T
    return abs(np.squeeze(np.matmul((np.matmul(x22,F)),x11)))

'''
 Function that computes the final fundamental matrix using RANSAC algorithm.
'''
def fundamental_Matrix(src_pts, dst_pts):

    # Get inliers RANSAC section 3.2.3 CMSC733 pseudo=-code
    M = 50  # iterations
    Total_inliers = 0
    FinalFundamentalMatrix = np.zeros((3,3))

    for i in range(M):

        rand_num_list = []
        Key_pts1 = []
        Key_pts2 = []
        TempKey_pts1 = []
        TempKey_pts2 = []

        count = 0
        # Select 8 random key points
        while len(rand_num_list) <= 8:
            rand_num = random.randint(0, len(src_pts) - 1)
            if rand_num not in rand_num_list:
                rand_num_list.append(rand_num)
        for index in rand_num_list:
            Key_pts1.append([src_pts[index][0], src_pts[index][1]])
            Key_pts2.append([dst_pts[index][0], dst_pts[index][1]])

        # Compute the fundamental matrix using the previously obtained 8 key points.
        F = calc_fundamental_mat(Key_pts1,Key_pts2)

        # Compute the error and calculate the total inliers for each F matrix.
        for ind in range(0, len(src_pts)):
            if CheckCondition(src_pts[ind], dst_pts[ind], F) < 0.01:
                count = count + 1
                TempKey_pts1.append(src_pts[ind])
                TempKey_pts2.append(dst_pts[ind])

        # Select the fundamental matrix that generates the maximum number of inliers.
        if count > Total_inliers:
            Total_inliers = count
            FinalFundamentalMatrix = F
            Final_Inliers1 = TempKey_pts1
            Final_Inliers2 = TempKey_pts2

    return FinalFundamentalMatrix, Final_Inliers1, Final_Inliers2


########################################################################################################################

###################################                        MAIN                          ###############################

########################################################################################################################


Translation = np.zeros((3, 1))
Rotation = np.eye(3)

H_Start = np.identity(4)
p_0 = np.array([[0, 0, 0, 1]]).T
flag = 0
frame_count = 0
data_points = []

path = r"C:\Users\kusha\Documents\ENPM673\PROJECT_3\ENPM673\Project_5\Oxford_dataset/stereo/centre"
list_of_filenames = [filename for filename in os.listdir(path)]
# print(list_of_filenames)
dataset_size = len(list_of_filenames)

fx, fy, c_x, c_y, camera_img, LUT = rd.ReadCameraModel(r'C:\Users\kusha\Documents\ENPM673\PROJECT_3\ENPM673\Project_5\Oxford_dataset/model')
K = np.array([[fx, 0, c_x], [0, fy, c_y], [0, 0, 1]])

for index in range(20, dataset_size - 1):
    frame_count = frame_count+1

    ####################################################################################################################
    #                                                      Load Image 1                                                #
    # STEPS:
    #   1: Read the first frame
    #   2: Convert the image from Bayer scale to BGR color scale
    #   3: Undistort the image using the given function
    #   4: Convert the image to Greyscale
    ####################################################################################################################

    img1 = cv2.imread(path + '/' + list_of_filenames[index], 0)

    colorimage1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
    undistortedimage1 = ud.UndistortImage(colorimage1, LUT)
    gray1 = cv2.cvtColor(undistortedimage1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread(path + '/' + list_of_filenames[index + 1], 0)
    colorimage2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
    undistortedimage2 = ud.UndistortImage(colorimage2, LUT)
    gray2 = cv2.cvtColor(undistortedimage2, cv2.COLOR_BGR2GRAY)

    grayImage1 = gray1[200:650, 0:1280]
    grayImage2 = gray2[200:650, 0:1280]

    # cv2.waitKey(0)

    ####################################################################################################################
    #                    Compute the Keypoints inbetween two consecutive frames using SIFT Algorithm                   #
    ####################################################################################################################

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(grayImage1, None)
    kp2, des2 = sift.detectAndCompute(grayImage2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    features1 = []
    features2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            features1.append(kp1[m.queryIdx].pt)
            features2.append(kp2[m.trainIdx].pt)

    Total_inliers = 0
    FinalFundamentalMatrix = np.zeros((3, 3))
    inlier1 = []
    inlier2 = []

    ####################################################################################################################
    #                                         Calculate Fundamental Matrix ( Using RANSAC)                             #
    ####################################################################################################################

    FinalFundamentalMatrix, inlier1, inlier2 = fundamental_Matrix(features1,features2)

    ####################################################################################################################
    #                                         Calculate Essential Matrix                                               #
    ####################################################################################################################

    E_matrix = estimate_Essential_Matrix(K, FinalFundamentalMatrix)

    ####################################################################################################################
    #                                         Estimate Camera Pose from Essential Matrix                               #
    ####################################################################################################################

    RotationMatrix, Tlist = cameraPose(E_matrix)

    ####################################################################################################################
    #                                         Obtain final Rotation and Translation using Linsear Triangulation.       #
    ####################################################################################################################

    finR, finC = linear_triangulation(RotationMatrix, Tlist, K, inlier1, inlier2)

    R = finR
    t = finC
    if t[2] < 0: t = -t

    ####################################################################################################################
    #                                         Plot the points and save the image from Visualization                    #
    ####################################################################################################################

    H_Start = H_Start @ Homogenousmatrix(R, t)
    p_projection = H_Start @ p_0

    print('x- ', p_projection[0])
    print('y- ', p_projection[2])
    data_points.append([-p_projection[0][0], p_projection[2][0]])
    plt.scatter(-p_projection[0][0], p_projection[2][0], color='r')

    print("FRAME COUNT:::", frame_count)
    # plt.savefig(r"C:\Users\kusha\Documents\ENPM673\PROJECT_3\ENPM673\Project_5\FinalOutput\output_" + str(frame_count) + ".png")
    plt.pause(0.1)

    if cv2.waitKey(1) == 27:
        break
    flag = flag + 1

cv2.destroyAllWindows()
plt.show()
