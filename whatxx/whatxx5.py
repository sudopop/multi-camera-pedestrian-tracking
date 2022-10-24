import cv2
import numpy as np
import time
import os

resizer = 0.1
OUTPUT = 'OUTPUT'


def ReadPhotos(Folder1, Folder2):
    try:
        os.mkdir(OUTPUT)
    except FileExistsError:
        print("Folder exists")
    JPEGlist = []
    TIFFlist = []
    if Folder1.endswith("jpeg") and Folder2.endswith("tiff") or Folder1.endswith("tiff") and Folder2.endswith("jpeg"):
        if Folder1.endswith("jpeg"):
            JPEGlist.append(Folder1)
            if Folder2.endswith(".tiff"):
                TIFFlist.append(Folder2)
        if Folder1.endswith("tiff"):
            JPEGlist.append(Folder2)
            if Folder2.endswith(".jpeg"):
                TIFFlist.append(Folder1)
    else:
        for files in os.listdir(Folder1):
            if files.endswith(".jpeg"):
                JPEGlist.append(Folder1 + "/" + files)
            if files.endswith(".tiff"):
                TIFFlist.append(Folder2 + "/" + files)
        for files in os.listdir(Folder2):
            if files.endswith(".jpeg"):
                JPEGlist.append(Folder2 + "/" + files)
            if files.endswith(".tiff"):
                TIFFlist.append(Folder2 + "/" + files)
    return JPEGlist, TIFFlist


def MainFunction(JPEGimage, TIFFimage, number=''):
    img_ = cv2.imread(JPEGimage)
    img_ = cv2.resize(img_, (0, 0), fx=resizer, fy=resizer)
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

    img = cv2.imread(TIFFimage)
    img = cv2.resize(img, (0, 0), fx=resizer, fy=resizer)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))

    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    '''
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       flags = 2)

    img3 = cv2.drawMatches(img_,kp1,img,kp2,good,None,**draw_params)
    cv2.imshow("original_image_drawMatches.jpg", img3)
    '''
    warped_image = cv2.warpPerspective(img_, M, (img.shape[1], img.shape[0]))

    RGBA_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2RGBA)
    RGBA_image[:, :, 3] = img2
    cv2.imwrite(OUTPUT + "/_" + str(number) + ".png", RGBA_image)


if __name__ == "__main__":
    start_time = time.time()

    RGB, TIFF = ReadPhotos('./data/sample/RGBimages', './data/sample/TIFFimages')
    for image in range(min(len(RGB), len(TIFF))):
        MainFunction(RGB[image], TIFF[image], image)
    print("Time took:", time.time() - start_time)
    print("Finished")