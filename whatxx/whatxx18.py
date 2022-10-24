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
    img_ceiling = cv2.imread('./data/sample/8_1.png', cv2.IMREAD_COLOR)
    img_ceiling = cv2.resize(img_ceiling, (960, 540))
    h, w, c = img_ceiling.shape

    img_angle = cv2.imread('./data/sample/8_2.png', cv2.IMREAD_COLOR)
    img_angle = cv2.resize(img_angle, (960, 540))

    #ceiling 탑뷰 호모그래피
    c_pts1 = np.float32([[257, 3], [1, 281], [473, 4], [546, 288]]) #돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    c_pts2 = np.float32([[0, 0], [0, 540], [960, 0], [960, 540]]) #이미지 전체 크기
    c_H = cv2.getPerspectiveTransform(c_pts1, c_pts2)
    img_top_ceil = cv2.warpPerspective(img_ceiling, c_H, (960, 540))

    # angle 탑뷰 호모그래피
    a_pts1 = np.float32([[364, 1], [77, 366], [663, 3], [957, 281]])  #돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_pts2 = np.float32([[0, 0], [0, 540], [960, 0], [960, 540]]) #이미지 전체 크기
    a_H = cv2.getPerspectiveTransform(a_pts1, a_pts2)
    img_top_angle = cv2.warpPerspective(img_angle, a_H, (960, 540))

    # angle 탑뷰 -> ceil 탑뷰 호모그래피 (겹치는 부분)
    c_top_pts = np.float32([[94, 330], [116, 538], [958, 260], [957, 539]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_top_pts = np.float32([[131, 121], [91, 338], [955, 13], [953, 287]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]턴
    ac_H = cv2.getPerspectiveTransform(a_top_pts, c_top_pts) #역함수에 사용할 경우 H (결과는 같음)
    ca_H = cv2.getPerspectiveTransform(c_top_pts, a_top_pts) #그냥 사용할 경우 H (결과는 같음)

    cv2.circle(img_ceiling, (257, 3), 3, (255, 0, 0), -1)
    cv2.circle(img_ceiling, (1, 281), 3, (0, 255, 0), -1)
    cv2.circle(img_ceiling, (473, 4), 3, (0, 0, 255), -1)
    cv2.circle(img_ceiling, (546, 288), 3, (255, 0, 255), -1)

    cv2.circle(img_angle, (364, 1), 3, (255, 0, 0), -1)
    cv2.circle(img_angle, (77, 366), 3, (0, 255, 0), -1)
    cv2.circle(img_angle, (663, 3), 3, (0, 0, 255), -1)
    cv2.circle(img_angle, (957, 281), 3, (255, 0, 255), -1)

    # 테스트: object 간편하게 수기입력 (xmin, ymin, xmax, ymax)
    # ceil_pig = [558, 228, 605, 353] #광
    # ceil_pig = [128, 293, 192, 440]  # 인턴
    ceil_pig = [258, 122, 295, 211]  #
    ceil_pig = [386, 121, 431, 246]  # 나
    ceil_to_topceil = transform_polygon(c_H, rectangle_to_polygon(ceil_pig))  # BB [626.22114081 185.16983677 742.21929358 328.59193591] #xmin, ymin, xmax, ymax
    top_ceil_to_top_angled = transform_polygon(ca_H, ceil_to_topceil)  # 역함수 대신 그냥 H로 ceil top-> angle top
    top_angled_to_angled = transform_polygon(np.linalg.inv(a_H), top_ceil_to_top_angled)
    print("test: ", top_angled_to_angled)
    img_result = cv2.rectangle(img_angle, (502, 24), (537, 72), (0, 255, 0), 5) #나
    img_result = cv2.rectangle(img_angle, (587, 26), (656, 76), (0, 255, 0), 5)  # 나

    # 이미지 좌표 따오기
    paste_coordinate = get_four_points(img_ceiling)
    print("paste_coordinate: ", paste_coordinate)
    paste_coordinate = np.array(paste_coordinate)


    # # img_result = cv2.rectangle(img_ceiling, (140, 189), (187, 297), (0, 255, 0), 2)
    #
    cv2.imshow('img_angle', img_angle)
    cv2.imshow('img_ceiling', img_ceiling)
    cv2.imshow('img_top_ceil', img_top_ceil)
    cv2.imshow('img_top_angle', img_top_angle)
    # cv2.imshow('test2', img_top_angle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()