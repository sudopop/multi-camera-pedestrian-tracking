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

    cv2.waitKey(150000)
    # while True:
    #     cv2.waitKey(1)
    #     if len(data) == 4:
    #         break

    points = np.array(data['points'], dtype=float)

    print(points)

    return points


if __name__ == '__main__':
    # Read source image
    img_ceiling = cv2.imread('./data/sample/1_1.png', cv2.IMREAD_COLOR)
    img_ceiling = cv2.resize(img_ceiling, (960, 540))
    h, w, c = img_ceiling.shape

    img_angle = cv2.imread('./data/sample/1_2.png', cv2.IMREAD_COLOR)
    img_angle = cv2.resize(img_angle, (960, 540))

    #ceiling 탑뷰 호모그래피
    c_pts1 = np.float32([[292, 3], [12, 404], [544, 0], [671, 483]]) #돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    c_pts2 = np.float32([[0, 0], [0, 540], [960, 0], [960, 540]]) #이미지 전체 크기
    c_H = cv2.getPerspectiveTransform(c_pts1, c_pts2)
    img_top_ceil = cv2.warpPerspective(img_ceiling, c_H, (960, 540))

    cv2.circle(img_ceiling, (292, 3), 3, (255, 0, 0), -1)
    cv2.circle(img_ceiling, (12, 404), 3, (0, 255, 0), -1)
    cv2.circle(img_ceiling, (544, 0), 3, (0, 0, 255), -1)
    cv2.circle(img_ceiling, (671, 483), 3, (0, 255, 255), -1)

    # angle 탑뷰 호모그래피
    a_pts1 = np.float32([[447, 1], [218, 344], [689, 4], [957, 286]])  #돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_pts2 = np.float32([[0, 0], [0, 540], [960, 0], [960, 540]]) #이미지 전체 크기
    a_H = cv2.getPerspectiveTransform(a_pts1, a_pts2)
    img_top_angle = cv2.warpPerspective(img_angle, a_H, (960, 540))

    cv2.circle(img_angle, (401, 1), 3, (255, 0, 0), -1)
    cv2.circle(img_angle, (111, 355), 3, (0, 255, 0), -1)
    cv2.circle(img_angle, (688, 2), 3, (0, 0, 255), -1)
    cv2.circle(img_angle, (955, 274), 3, (0, 255, 255), -1)

    ##이미지 좌표 따오기
    paste_coordinate = get_four_points(img_ceiling)
    print("paste_coordinate: ", paste_coordinate)
    paste_coordinate = np.array(paste_coordinate)

    # angle 탑뷰 -> ceil 탑뷰 호모그래피 (겹치는 부분)
    c_top_pts = np.float32([[0, 203], [11, 533], [956, 200], [957, 537]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_top_pts = np.float32([[126.,  93.], [ 90., 344.], [956.,  24.], [955., 331.]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    ac_H = cv2.getPerspectiveTransform(a_top_pts, c_top_pts) #역함수에 사용할 경우 H (결과는 같음)
    ca_H = cv2.getPerspectiveTransform(c_top_pts, a_top_pts) #그냥 사용할 경우 H (결과는 같음)

    # 테스트: object 간편하게 수기입력 (xmin, ymin, xmax, ymax)
    ceil_pig = [329, 81, 370, 211]
    # ceil_pig = [115, 171, 162, 304]
    # ceil_pig = [378, 269, 447, 422]
    ceil_to_topceil = transform_polygon(c_H, rectangle_to_polygon(ceil_pig))  # BB [626.22114081 185.16983677 742.21929358 328.59193591] #xmin, ymin, xmax, ymax
    # top_ceil_to_top_angled = transform_polygon(np.linalg.inv(ac_H), ceil_to_topceil)  # 역함수 H로 top_angle -> top_ceil로 변환하는데 사용한 값
    top_ceil_to_top_angled = transform_polygon(ca_H, ceil_to_topceil)  # 역함수 대신 그냥 H로 ceil top-> angle top
    top_angled_to_angled = transform_polygon(np.linalg.inv(a_H), top_ceil_to_top_angled)
    print("test: ", top_angled_to_angled)
    # img_result = cv2.rectangle(img_angle, top_angled_to_angled['tl'], top_angled_to_angled['br'], (0, 255, 0), 5)
    img_result = cv2.rectangle(img_angle, (603, 69), (663, 102), (0, 255, 0), 2)
    img_result = cv2.rectangle(img_angle, (409, 55), (475, 89), (0, 255, 0), 2)
    img_result = cv2.rectangle(img_angle, (536, 17), (588, 54), (0, 255, 0), 2)
    #
    cv2.imshow('img_angle', img_angle)
    cv2.imshow('img_ceiling', img_ceiling)
    cv2.imshow('img_top_ceil', img_top_ceil)
    cv2.imshow('img_top_angle', img_top_angle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





