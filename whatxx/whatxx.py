# import cv2
# import numpy as np
#
# img = cv2.imread("./data/sample/ceiling.png")
#
# frame = cv2.resize(img, (640, 480))
#
# tl = (222, 387)
# bl = (70, 472)
# tr = (400, 380)
# br = (538, 472)
#
# cv2.circle(frame, tl, 5, (0,0,255), -1)
# cv2.circle(frame, bl, 5, (0,0,255), -1)
# cv2.circle(frame, tr, 5, (0,0,255), -1)
# cv2.circle(frame, br, 5, (0,0,255), -1)
#
# # pts1 = [tl, br, tr, br]
# pts1 = np.float32([[222,387],[70,472],[400,380],[538,472]])
# # pts2 = [[0,0], [0,480], [640,0], [640,480]]
# pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])
#
# matrix =cv2.getPerspectiveTransform(pts1, pts2)
# M = cv2.getPerspectiveTransform(pts1, pts2)
#
# cv2.imshow("Frame", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread("./data/sample/ceiling.png")
# # [x,y] 좌표점을 4x2의 행렬로 작성
# # 좌표점은 좌상->좌하->우상->우하
# pts1 = np.float32([[504,1003],[243,1525],[1000,1000],[1280,1685]])
#
# # 좌표의 이동점
# pts2 = np.float32([[10,10],[10,1000],[1000,10],[1000,1000]])
#
# # pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
# cv2.circle(img, (504,1003), 20, (255,0,0),-1)
# cv2.circle(img, (243,1524), 20, (0,255,0),-1)
# cv2.circle(img, (1000,1000), 20, (0,0,255),-1)
# cv2.circle(img, (1280,1685), 20, (0,0,0),-1)
#
# M = cv2.getPerspectiveTransform(pts1, pts2)
#
# dst = cv2.warpPerspective(img, M, (1100,1100))
#
# plt.subplot(121),plt.imshow(img),plt.title('image')
# plt.subplot(122),plt.imshow(dst),plt.title('Perspective')
# plt.show()

import numpy as np
import cv2
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
# img1 = cv2.imread('./data/sample/ceiling.png',0) # queryImage
img1 = cv2.imread('./data/sample/angle.png',0) # trainImage
img2 = cv2.imread('./data/sample/angle.png',0) # trainImage

img1 = cv2.resize(img1, (640, 480))
img2 = cv2.resize(img2, (640, 480))

# Initiate SIFT detector
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
    if m.distance < 0.3*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()