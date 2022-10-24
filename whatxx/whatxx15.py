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

    #ceiling 탑뷰 호모그래피
    c_pts1 = np.float32([[279, 13], [11, 408], [554, 19], [667, 480]]) #돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    c_pts2 = np.float32([[0, 0], [0, 540], [960, 0], [960, 540]]) #이미지 전체 크기
    c_H = cv2.getPerspectiveTransform(c_pts1, c_pts2)
    img_top_ceil = cv2.warpPerspective(img_ceiling, c_H, (960, 540))

    # angle 탑뷰 호모그래피
    a_pts1 = np.float32([[431, 12], [206, 374], [694, 5], [959, 303]])  #돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_pts2 = np.float32([[0, 0], [0, 540], [960, 0], [960, 540]]) #이미지 전체 크기
    a_H = cv2.getPerspectiveTransform(a_pts1, a_pts2)
    img_top_angle = cv2.warpPerspective(img_angle, a_H, (960, 540))

    # angle 탑뷰 -> ceil 탑뷰 호모그래피 (겹치는 부분)
    c_top_pts = np.float32([[1, 190], [1, 537], [956, 161], [955, 539]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_top_pts = np.float32([[8, 34], [6, 333], [955, 14], [953, 289]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]턴
    ac_H = cv2.getPerspectiveTransform(a_top_pts, c_top_pts) #역함수에 사용할 경우 H (결과는 같음)
    ca_H = cv2.getPerspectiveTransform(c_top_pts, a_top_pts) #그냥 사용할 경우 H (결과는 같음)

    cv2.circle(img_top_ceil, (1, 190), 3, (255, 0, 0), -1)
    cv2.circle(img_top_ceil, (1, 537), 3, (0, 255, 0), -1)
    cv2.circle(img_top_ceil, (956, 161), 3, (0, 0, 255), -1)
    cv2.circle(img_top_ceil, (955, 539), 3, (0, 0, 0), -1)

    cv2.circle(img_top_angle, (8, 34), 3, (255, 0, 0), -1)
    cv2.circle(img_top_angle, (6, 333), 3, (0, 255, 0), -1)
    cv2.circle(img_top_angle, (955, 14), 3, (0, 0, 255), -1)
    cv2.circle(img_top_angle, (953, 289), 3, (0, 0, 0), -1)

    # 테스트: object 간편하게 수기입력 (xmin, ymin, xmax, ymax)
    # ceil_pig = [427, 132, 484, 281] #나
    # ceil_pig = [113, 283, 193, 429]  # 인턴
    ceil_pig = [110, 145, 165, 263]  # 준광
    ceil_to_topceil = transform_polygon(c_H, rectangle_to_polygon(ceil_pig))  # BB [626.22114081 185.16983677 742.21929358 328.59193591] #xmin, ymin, xmax, ymax
    top_ceil_to_top_angled = transform_polygon(ca_H, ceil_to_topceil)  # 역함수 대신 그냥 H로 ceil top-> angle top
    top_angled_to_angled = transform_polygon(np.linalg.inv(a_H), top_ceil_to_top_angled)
    print("test: ", top_angled_to_angled)
    img_result = cv2.rectangle(img_angle, (397, 93), (513, 121), (0, 255, 0), 5) #나
    img_result = cv2.rectangle(img_angle, (325, 48), (438, 82), (0, 255, 0), 5)  # 나
    # img_result = cv2.rectangle(img_angle, (375, 121), (486, 178), (0, 255, 0), 5) #인
    # img_result = cv2.rectangle(img_angle, (332, 57), (417, 108), (0, 255, 0), 5)  # 인
    # img_result = cv2.rectangle(img_angle, (607, 22), (673, 60), (0, 255, 0), 5)  # 나
    # img_result = cv2.rectangle(img_angle, (407, 79), (507, 115), (0, 255, 0), 5)  # 나
    # img_result = cv2.rectangle(img_angle, (398, 36), (412, 76), (0, 255, 0), 5)  # 나

    # ##이미지 좌표 따오기
    # paste_coordinate = get_four_points(img_top_angle)
    # print("paste_coordinate: ", paste_coordinate)
    # paste_coordinate = np.array(paste_coordinate)


    # # img_result = cv2.rectangle(img_ceiling, (140, 189), (187, 297), (0, 255, 0), 2)
    #
    cv2.imshow('test', img_result)
    # cv2.imshow('test2', img_top_angle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





