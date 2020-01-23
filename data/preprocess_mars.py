import os
import cv2
import numpy as np
import random
import pickle

def get_all_file_names(dirName):
    # Get the list of all files in directory tree at given path
    listOfFilesLeft = list()
    listOfFilesRight = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        # listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        for file in filenames:
            if file.endswith("left.ppm"):
                listOfFilesLeft.append(os.path.join(dirpath, file))
            elif file.endswith("right.ppm"):
                listOfFilesRight.append(os.path.join(dirpath, file))
    
    return listOfFilesLeft, listOfFilesRight

def stitch_image(right_image, left_image, final_name):
    img_ = cv2.imread(right_image)
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    # img1 = cv2.convertScaleAbs(img1, alpha=1.0, beta=100)

    img = cv2.imread(left_image)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img2 = cv2.convertScaleAbs(img2, alpha=1.0, beta=100)

    sift = cv2.xfeatures2d.SIFT_create()
    # find key points
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    #cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))
    #FLANN_INDEX_KDTREE = 0
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks = 50)
    #match = cv2.FlannBasedMatcher(index_params, search_params)
    
    # match = cv2.BFMatcher()
    # matches = match.knnMatch(des1,des2,k=2)

    match = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = match.match(des1,des2)
    # print(matches)
    matches = sorted(matches, key = lambda x:x.distance)
    good = []
    for m in matches:
        good.append(m)
        if len(good) > 10:
            break


    # good = []
    # for m,n in matches:
    #     if m.distance < 0.03*n.distance:
    #         good.append(m)

    draw_params = dict(matchColor=(0,255,0),
                        singlePointColor=None,
                        flags=2)
    img3 = cv2.drawMatches(img_,kp1,img,kp2,good,None,**draw_params)

    # while(1):
    #     cv2.imshow("original_image_drawMatches.jpg", img3)
    #     k = cv2.waitKey(33)
    #     if k==27:    # Esc key to stop
    #         break
    #     elif k==-1:  # normally -1 returned,so don't print it
    #         continue
    #     else:
    #         print(k) # else print its value
    
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        #cv2.imshow("original_image_overlapping.jpg", img2)
    else:
        print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))
        # return

    dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0],0:img.shape[1]] = img
    # cv2.imshow("original_image_stitched.jpg", dst)
    def trim(frame):
        #crop top
        if not np.sum(frame[0]):
            return trim(frame[1:])
        #crop top
        if not np.sum(frame[-1]):
            return trim(frame[:-2])
        #crop top
        if not np.sum(frame[:,0]):
            return trim(frame[:,1:])
        #crop top
        if not np.sum(frame[:,-1]):
            return trim(frame[:,:-2])

        crop_img = frame[0:1000, 0:1000]
        
        return crop_img
    # cv2.imshow("original_image_stitched_crop.jpg", trim(dst))

    # file_name = "C:/Users/rizwan.asif/Desktop/projects/SfMLearner/formatted_mars_data/" + str(final_name) + ".jpg"
    file_name = os.path.join(os.path.abspath("../formatted_mars_data/"), str(final_name) + ".jpg")
    file_name = file_name.replace("\\", "/")
    cv2.imwrite( file_name, trim(dst))
    # print("saved ", i)
    return file_name

left, right = get_all_file_names("../mars_data")

assert(len(left) == len(right))

file_names = list()

for i in range(len(left)):
    name = stitch_image(right[i], left[i], i)
    file_names.append(name)
print(file_names)

SPLIT_RATIO = 0.2

random.shuffle(file_names)
split = int(len(file_names) * SPLIT_RATIO)

train = file_names[:len(file_names)-split]
test = file_names[len(file_names)-split:]

with open("train.txt", "wb") as fp:
    pickle.dump(train, fp)

with open("test.txt", "wb") as fp:
    pickle.dump(test, fp)