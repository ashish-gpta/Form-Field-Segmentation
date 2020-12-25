#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import matplotlib
from scipy import ndimage
matplotlib.rcParams['figure.figsize'] = [30, 20]


# In[89]:

for i in range(1,15):
    MIN_MATCH_COUNT = 20

    img1 = cv2.imread('test.jpg',0)          # queryImage
    img2 = cv2.imread('{}.jpg'.format(i),0) # trainImage


    # In[90]:


    img_edges = cv2.Canny(img2, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    angles = []

    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 4)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    img2 = ndimage.rotate(img2, median_angle,order=3)


    # In[91]:


    # Initiate SIFT detector
    # sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    # In[92]:


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        print(dst)
    #     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,30, cv2.LINE_AA)
        print(dst[1][0][1])
        x_cord = int(dst[0][0][0])
        x_length = int(dst[3][0][0]-dst[0][0][0]-dst[1][0][0]+dst[2][0][0])//2
        y_cord = int(dst[0][0][1])
        y_length = int(dst[2][0][1]-dst[0][0][1]-dst[3][0][1]+dst[1][0][1])//2
    #     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        cropped = img2[y_cord:y_cord+y_length,x_cord:x_cord+x_length]
    else:
        # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None


    # In[93]:


    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                    singlePointColor = None,
    #                    matchesMask = matchesMask, # draw only inliers
    #                    flags = 2)

    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    #plt.imshow(img2, 'gray'),plt.show()


    # In[94]:


    #plt.imshow(cropped, 'gray'),plt.show()


    # In[95]:


    resized = cv2.resize(cropped,(1750,1250))


    # In[96]:


    #plt.imshow(resized, 'gray'),plt.show()


    # In[97]:


    equ = cv2.equalizeHist(resized)
    #plt.imshow(equ, 'gray'),plt.show()


    # In[98]:


    img = cv2.medianBlur(equ,5)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,            cv2.THRESH_BINARY,51,0)
    #plt.imshow(th2, 'gray'),plt.show()


    # In[99]:


    new=cv2.erode(th2,cv2.getStructuringElement(cv2.MORPH_RECT,(1,5)),iterations = 1).astype(np.uint8)
    # new=cv2.erode(new,cv2.getStructuringElement(cv2.MORPH_RECT,(3,1)),iterations = 4).astype(np.uint8)
    #plt.imshow(new, 'gray'),plt.show() 


    # In[100]:


    connectivity = 4  
    output = cv2.connectedComponentsWithStats(new, connectivity, cv2.CV_32S)
    newI=np.copy(new)
    for i in range(output[0]):
        if output[2][i,4]<3000 or output[2][i,3]>500 or output[2][i,2]>2000:
            newI[output[1]==i]=0


    # In[101]:


    #plt.imshow(newI, 'gray'),plt.show() 


    # In[102]:


    final=np.copy(255-newI)


    # In[103]:


    output2 = cv2.connectedComponentsWithStats(final, connectivity, cv2.CV_32S)


    # In[104]:


    for i in range(output2[0]):
        if output2[2][i,4]<1500:
            final[output2[1]==i]=0
    final2=np.copy(255-final)


    # In[105]:


    #plt.imshow(final2, 'gray'),plt.show()  


    # In[106]:


    final2=np.copy(255-final)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(20,25))
    mask = cv2.morphologyEx(final2, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(mask, 'gray'),plt.show()  


    # In[107]:


    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT,(15,5)),iterations = 1).astype(np.uint8)
    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT,(3,2)),iterations = 1).astype(np.uint8)

    #plt.imshow(mask, 'gray'),plt.show()  


    # In[108]:


    #plt.imshow((255-np.multiply(mask/255,resized)), 'gray'),plt.show()  


    # In[109]:


    masked=(255-np.multiply(mask/255,resized))
    masked=np.where(masked==255,0,masked)
    #plt.imshow(masked, 'gray'),plt.show() 


    # In[112]:


    # masked_Hist= cv2.equalizeHist(masked)
    # #plt.imshow(masked_Hist, 'gray'),plt.show()  


    # In[115]:


    # masked_Dil=cv2.dilate(masked_Hist,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations = 1)
    masked_Dil=cv2.dilate(masked,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations = 1)
    #plt.imshow(masked_Dil, 'gray'),plt.show()  


    # In[126]:


    img = cv2.medianBlur(masked_Dil,5)
    # img = cv2.medianBlur(masked,5)
    ret,th1 = cv2.threshold(img,110,255,cv2.THRESH_BINARY) #170 for booklet
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,            cv2.THRESH_BINARY,101,0)
    #plt.imshow(th1, 'gray'),plt.show()  


    # In[153]:


    output3 = cv2.connectedComponentsWithStats(th1, connectivity, cv2.CV_32S)
    newI=np.copy(th1)
    mx=0
    for i in range(output3[0]):
        if output3[2][i,4]<50 or output3[2][i,4]>2000 or output3[2][i,3]>50 or output3[2][i,2]>200 or output3[2][i,3]<17 or (float)(output3[2][i,2])/(float)(output3[2][i,3])>3:
            newI[output3[1]==i]=0


    # In[154]:


    #plt.imshow(newI, 'gray'),plt.show()  

    cv2.imwrite('{}p.jpg'.format(i),newI)
# In[150]:


# output3 = cv2.connectedComponentsWithStats(newI, connectivity, cv2.CV_32S)
# newI=np.copy(newI)
# mx=0
# for i in range(output3[0]):
# #     print((output3[2][i,3]),(output3[2][i,2]))
# #     print((float)(output3[2][i,2])/(float)(output3[2][i,3]))
#     if (float)(output3[2][i,2])/(float)(output3[2][i,3])>3:
#         newI[output3[1]==i]=0
# #plt.imshow(newI, 'gray'),plt.show()  


# In[151]:


# cv2.imwrite('44p.jpg',newI)


# In[ ]:




