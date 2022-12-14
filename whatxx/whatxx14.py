#https://medium.com/@conghung43/image-projective-transformation-with-opencv-python-f0028aaf2b6d
#copyto: https://deep-learning-study.tistory.com/104

import cv2 as cv2
import numpy as np

def transform_polygon(H, poly_dict):
    """
    H : 3x3 matrix
    poly_dict : {"tl": [x1, y1], ...}

    new_poly_dict : {"tl": [H(x1, y1)], ...}
    """
    new_poly_dict = {}
    for pos, old_p in poly_dict.items():
        new_p = np.matmul(H, np.array(old_p + [1]))
        new_poly_dict[pos] = [int(new_p[0] / new_p[2]), int(new_p[1] / new_p[2])]
    return new_poly_dict


def rectangle_to_polygon(corners):
    xmin, ymin, xmax, ymax = corners
    poly_dict = {
        "tl": [xmin, ymin],
        "tr": [xmax, ymin],
        "br": [xmax, ymax],
        "bl": [xmin, ymax]
    }
    return poly_dict


# This function will get click pixel coordinate that source image will be pasted to destination image
def get_paste_position(event, x, y, flags, paste_coordinate_list):
    cv2.imshow('collect coordinate', img_dest_copy)
    if event == cv2.EVENT_LBUTTONUP:
    # Draw circle right in click position
        cv2.circle(img_dest_copy, (x, y), 2, (0, 0, 255), -1)
        # Append new clicked coordinate to paste_coordinate_list
        paste_coordinate_list.append([x, y])

def mouse_handler(event, x, y, flasgs, data):
    if event== cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['im'], (x,y),3,(0,0,255), -1)
        cv2.imshow('image', data['im'])

        if len(data['points']) < 4 :
            data['points'].append([x,y])


def get_four_points(im):
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    cv2.imshow('image', im)
    cv2.setMouseCallback('image', mouse_handler, data)

    cv2.waitKey(45000)
    # while True:
    #     cv2.waitKey(1)
    #     if len(data) == 4:
    #         break

    points = np.array(data['points'], dtype=float)

    print(points)

    return points


if __name__ == '__main__':
    # Read source image
    img_ceiling = cv2.imread('./data/sample/2_1.png', cv2.IMREAD_COLOR)
    img_ceiling = cv2.resize(img_ceiling, (960, 540))
    h, w, c = img_ceiling.shape

    img_angle = cv2.imread('./data/sample/2_2.png', cv2.IMREAD_COLOR)
    img_angle = cv2.resize(img_angle, (960, 540))

    #ceiling ?????? ???????????????
    c_pts1 = np.float32([[268, 2], [  2., 410.], [551., 0.], [666., 469.]]) #???????????? ????????? ?????? [[left,top], [left,bottom], [right, top], [right, bottom]]
    c_pts2 = np.float32([[0, 0], [0, 540], [960, 0], [960, 540]]) #????????? ?????? ??????
    c_H = cv2.getPerspectiveTransform(c_pts1, c_pts2)
    img_top_ceil = cv2.warpPerspective(img_ceiling, c_H, (960, 540))

    cv2.circle(img_ceiling, (268, 2), 3, (255, 0, 0), -1)
    cv2.circle(img_ceiling, ( 2, 410), 3, (0, 255, 0), -1)
    cv2.circle(img_ceiling, (551, 0), 3, (0, 0, 255), -1)
    cv2.circle(img_ceiling, (666, 469), 3, (0, 255, 255), -1)

    # angle ?????? ???????????????
    a_pts1 = np.float32([[420., 1.], [147., 363.], [686., 0.], [959., 302.]])  #???????????? ????????? ?????? [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_pts2 = np.float32([[0, 0], [0, 540], [960, 0], [960, 540]]) #????????? ?????? ??????
    a_H = cv2.getPerspectiveTransform(a_pts1, a_pts2)
    img_top_angle = cv2.warpPerspective(img_angle, a_H, (960, 540))

    cv2.circle(img_angle, (420, 1), 3, (255, 0, 0), -1)
    cv2.circle(img_angle, (147, 363), 3, (0, 255, 0), -1)
    cv2.circle(img_angle, (686, 0), 3, (0, 0, 255), -1)
    cv2.circle(img_angle, (959, 302), 3, (0, 255, 255), -1)

    # angle ?????? -> ceil ?????? ??????????????? (????????? ??????)
    c_top_pts = np.float32([[ 2.,  197.], [ 13., 536.], [958.,  188.], [958., 538.]])  # ???????????? ????????? ?????? [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_top_pts = np.float32([[84, 90], [ 58., 352.], [958.,  37.], [956., 300.]])  # ???????????? ????????? ?????? [[left,top], [left,bottom], [right, top], [right, bottom]]???
    ac_H = cv2.getPerspectiveTransform(a_top_pts, c_top_pts) #???????????? ????????? ?????? H (????????? ??????)
    ca_H = cv2.getPerspectiveTransform(c_top_pts, a_top_pts) #?????? ????????? ?????? H (????????? ??????)

    # ?????????: object ???????????? ???????????? (xmin, ymin, xmax, ymax)
    ceil_pig = [435, 132, 472, 279] #???
    ceil_pig = [113, 283, 193, 429]  # ??????
    # ceil_pig = [110, 145, 165, 263]  # ??????
    ceil_to_topceil = transform_polygon(c_H, rectangle_to_polygon(ceil_pig))  # BB [626.22114081 185.16983677 742.21929358 328.59193591] #xmin, ymin, xmax, ymax
    # top_ceil_to_top_angled = transform_polygon(np.linalg.inv(ac_H), ceil_to_topceil)  # ????????? H??? top_angle -> top_ceil??? ??????????????? ????????? ???
    top_ceil_to_top_angled = transform_polygon(ca_H, ceil_to_topceil)  # ????????? ?????? ?????? H??? ceil top-> angle top
    top_angled_to_angled = transform_polygon(np.linalg.inv(a_H), top_ceil_to_top_angled)
    print("test: ", top_angled_to_angled)
    img_result = cv2.rectangle(img_angle, (612, 26), (655, 63), (0, 255, 0), 5) #???
    img_result = cv2.rectangle(img_angle, (400, 90), (500, 119), (0, 255, 0), 5) #???
    img_result = cv2.rectangle(img_angle, (352, 48), (439, 80), (0, 255, 0), 5)  # ???

    # ##????????? ?????? ?????????
    paste_coordinate = get_four_points(img_ceiling)
    print("paste_coordinate: ", paste_coordinate)
    paste_coordinate = np.array(paste_coordinate)


    # # img_result = cv2.rectangle(img_ceiling, (140, 189), (187, 297), (0, 255, 0), 2)
    #
    cv2.imshow('img_angle', img_angle)
    cv2.imshow('img_ceiling', img_ceiling)
    cv2.imshow('img_top_ceil', img_top_ceil)
    cv2.imshow('img_top_angle', img_top_angle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





