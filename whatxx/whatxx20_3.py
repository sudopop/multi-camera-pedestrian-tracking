#https://medium.com/@conghung43/image-projective-transformation-with-opencv-python-f0028aaf2b6d
#copyto: https://deep-learning-study.tistory.com/104

import cv2 as cv2
import numpy as np
import pickle

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

    points = np.array(data['points'], dtype=int)

    print(points*2)

    return points


if __name__ == '__main__':
    # Read source image
    img_ceiling = cv2.imread('./sample/13_1.png', cv2.IMREAD_COLOR)
    cv2.circle(img_ceiling, (480, 288), 20, (255, 0, 0), -1)
    cv2.circle(img_ceiling, (122, 1074), 20, (0, 255, 0), -1)
    cv2.circle(img_ceiling, (1178, 242), 20, (0, 0, 255), -1)
    cv2.circle(img_ceiling, (1556, 1068), 20, (0, 255, 255), -1)
    h, w, c = img_ceiling.shape

    img_angle = cv2.imread('./sample/13_2.png', cv2.IMREAD_COLOR)
    cv2.circle(img_angle, (818, 24), 20, (255, 0, 0), -1)
    cv2.circle(img_angle, (338, 812), 20, (0, 255, 0), -1)
    cv2.circle(img_angle, (1340, 6), 20, (0, 0, 255), -1)
    cv2.circle(img_angle, (1916, 614), 20, (0, 255, 255), -1)
    img_ceiling = cv2.resize(img_ceiling, (960, 540))
    img_angle = cv2.resize(img_angle, (960, 540))

    cv2.imshow('img_ceiling', img_ceiling)
    cv2.imwrite("save/img_ceiling.png", img_ceiling)

    #ceiling 탑뷰 호모그래피
    c_pts1 = np.float32([[ 480,  288], [ 122, 1074], [1178,  242], [1556, 1068]]) #돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    c_pts2 = np.float32([[0, 0], [0, 1080], [1920, 0], [1920, 1080]]) #이미지 전체 크기
    c_H = cv2.getPerspectiveTransform(c_pts1, c_pts2)
    img_ceiling = cv2.resize(img_ceiling, (1920, 1080))
    img_top_ceil = cv2.warpPerspective(img_ceiling, c_H, (1920, 1080))

    # angle 탑뷰 호모그래피
    a_pts1 = np.float32([[ 818, 24], [ 338, 812], [1340, 6], [1916, 614]])  #돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_pts2 = np.float32([[0, 0], [0, 1080], [1920, 0], [1920, 1080]]) #이미지 전체 크기
    a_H = cv2.getPerspectiveTransform(a_pts1, a_pts2)
    img_angle = cv2.resize(img_angle, (1920, 1080))
    img_top_angle = cv2.warpPerspective(img_angle, a_H, (1920, 1080))

    # # # #이미지 좌표 따오기
    # img_target = img_top_ceil
    # img_target = cv2.resize(img_target, (960, 540))
    # img_target = get_four_points(img_target)

    # angle 탑뷰 -> ceil 탑뷰 호모그래피 (겹치는 부분)
    c_top_pts = np.float32([[6, 14], [6, 896], [1858, 10], [1880, 924]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_top_pts = np.float32([[12, 70], [2, 654], [1908, 8], [1916, 612]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]턴
    ac_H = cv2.getPerspectiveTransform(a_top_pts, c_top_pts) #역함수에 사용할 경우 H (결과는 같음)
    ca_H = cv2.getPerspectiveTransform(c_top_pts, a_top_pts) #그냥 사용할 경우 H (결과는 같음)

    # 테스트: object 간편하게 수기입력 (xmin, ymin, xmax, ymax)
    ceil_pig = [340, 790, 538, 1072]
    # ceil_pig = [1168, 826, 1264, 1068]

    ceil_to_topceil = transform_polygon(c_H, rectangle_to_polygon(ceil_pig))  # BB [626.22114081 185.16983677 742.21929358 328.59193591] #xmin, ymin, xmax, ymax
    top_ceil_to_top_angled = transform_polygon(ca_H, ceil_to_topceil)  # 역함수 대신 그냥 H로 ceil top-> angle top
    top_angled_to_angled = transform_polygon(np.linalg.inv(a_H), top_ceil_to_top_angled)
    print("ceil_pig: ", top_angled_to_angled['tl'][1])

    # img_result = cv2.rectangle(img_angle, (757, 245), (956, 326), (0, 255, 0), 5)
    img_result = cv2.rectangle(img_angle, (top_angled_to_angled['tl'][0], top_angled_to_angled['tl'][1]), (top_angled_to_angled['br'][0], top_angled_to_angled['br'][1]), (0, 255, 0), 5)

    with open('data_v3.pickle', 'wb') as f:
        print(c_H)
        pickle.dump(c_H, f, pickle.HIGHEST_PROTOCOL)
        print(ca_H)
        pickle.dump(ca_H, f, pickle.HIGHEST_PROTOCOL)
        print(a_H)
        pickle.dump(a_H, f, pickle.HIGHEST_PROTOCOL)

    # # img_result = cv2.rectangle(img_ceiling, (140, 189), (187, 297), (0, 255, 0), 2)
    #
    img_angle = cv2.resize(img_angle, (960, 540))
    img_ceiling = cv2.resize(img_ceiling, (960, 540))
    img_top_ceil = cv2.resize(img_top_ceil, (960, 540))
    img_top_angle = cv2.resize(img_top_angle, (960, 540))

    # cv2.imshow('img_angle', img_angle)
    # cv2.imwrite("save/img_angle.png", img_angle)
    # cv2.imshow('img_ceiling', img_ceiling)
    # cv2.imwrite("save/img_ceiling.png", img_ceiling)
    # cv2.imshow('img_top_ceil', img_top_ceil)
    # cv2.imwrite("save/img_top_ceil.png", img_top_ceil)
    # cv2.imshow('img_top_angle', img_top_angle)
    # cv2.imwrite("save/img_top_angle.png", img_top_angle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





