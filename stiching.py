import numpy as np
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
cv2.ocl.setUseOpenCL(False)
f_e_algo = 'sift'
f_t_match = 'br' #bruteforce
lst=["/home/jerry/Desktop/stiching/1.jpg","/home/jerry/Desktop/stiching/2.jpg","/home/jerry/Desktop/stiching/3.jpg","/home/jerry/Desktop/stiching/4.jpg"]
p1=cv2.imread(lst[0])
# cv2.imshow("pta nhi",p1)
# cv2.waitKey(0)
p1_RGB=cv2.cvtColor(p1,cv2.COLOR_BGR2RGB) #matplotlib RGB me hai aur OpenCV BGR me
pv1_gray = cv2.cvtColor(p1_RGB,cv2.COLOR_RGB2GRAY)
p2=cv2.imread(lst[1])
p2_RGB=cv2.cvtColor(p2,cv2.COLOR_BGR2RGB)
pv2_gray = cv2.cvtColor(p2_RGB,cv2.COLOR_RGB2GRAY)
np1=np.array(p1)
np2=np.array(p2)

def select_method(image, method):
    desc=cv2.SIFT_create()
    (keypoints, features) = desc.detectAndCompute(image, None)
    return (keypoints, features)

def kp_f():
    global keypoints_p1
    global keypoints_p2
    global feature_p1
    global feature_p2
    keypoints_p1,feature_p1= select_method(np1,method= f_e_algo)
    keypoints_p2,feature_p2= select_method(np2,method= f_e_algo)
kp_f()

def c_m_object(method, crossCheck):
    bf= cv2.BFMatcher(cv2.NORM_L2, crossCheck = crossCheck)
    return bf

def k_p_matching(feature_p1,feature_p2,method):
    bf=c_m_object(method,crossCheck=True)
    best_matches= bf.match(feature_p1,feature_p2)
    raw_matches=sorted(best_matches, key= lambda x: x.distance)
    return raw_matches

fig2= plt.figure(figsize=(20,8))
if f_t_match == 'br':
    matches = k_p_matching(feature_p1,feature_p2,method=f_e_algo)
    m_f_image = cv2.drawMatches(p1,keypoints_p1,p2,keypoints_p2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def homo_stiching(keypoints_p1,keypoints_p2,matches,reprojThres):
    keypoints_p1=np.float32([keypoint.pt for keypoint in keypoints_p1])
    keypoints_p2=np.float32([keypoint.pt for keypoint in keypoints_p2])
    if len(matches) >4:
        point_p1=np.float32([keypoints_p1[m.queryIdx] for m in matches])
        point_p2=np.float32([keypoints_p2[m.trainIdx] for m in matches])
        (M,status) = cv2.findHomography(point_p1,point_p2,cv2.RANSAC,reprojThres)
        return (matches, M, status)
    else:
        return None

M=homo_stiching(keypoints_p1,keypoints_p2,matches,reprojThres=4)
(matches,homo_matrix,status) = M

width = p2.shape[1] + p1.shape[1]
height = max(p2.shape[0],p1.shape[0])

result = cv2.warpPerspective(p1,homo_matrix, (width,height))
result[0:p2.shape[0], 0:p2.shape[1]] = p2



for i in range(len(lst)-2):
    f=cv2.imread(lst[i+2])
# cv2.imshow("pta nhi",p1)
# cv2.waitKey(0)
    f_RGB=cv2.cvtColor(f,cv2.COLOR_BGR2RGB) #matplotlib RGB me hai aur OpenCV BGR me
    fv1_gray = cv2.cvtColor(f_RGB,cv2.COLOR_RGB2GRAY)
    nf=np.array(f)
    keypoints_f,feature_f= select_method(nf,method= f_e_algo)
    result_RGB=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
    resultv2_gray = cv2.cvtColor(result_RGB,cv2.COLOR_RGB2GRAY)
    nresult=np.array(result)
    keypoints_result,feature_result= select_method(nresult,method= f_e_algo)
    
    
    if f_t_match=='br':
        matches_2 = k_p_matching(feature_f,feature_result,method=f_e_algo)
        m_f_image = cv2.drawMatches(f,keypoints_result,result,keypoints_result,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    M=homo_stiching(keypoints_f,keypoints_result,matches_2,reprojThres=4)
    (matches_2,homo_matrix,status) = M

    width = result.shape[1] + f.shape[1]
    height = max(result.shape[0],f.shape[0])

    result = cv2.warpPerspective(f,homo_matrix, (width,height))
    result[0:result.shape[0], 0:result.shape[1]] = result
cv2.imshow("ho gya",result)
cv2.waitKey(0)