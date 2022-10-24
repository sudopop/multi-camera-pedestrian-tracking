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

    cv2.waitKey(10000)
    # while True:
    #     cv2.waitKey(1)
    #     if len(data) == 4:
    #         break

    points = np.array(data['points'], dtype=float)

    print(points)

    return points


if __name__ == '__main__':
    # Read source image
    img_ceiling = cv2.imread('./data/sample/ceiling2.png', cv2.IMREAD_COLOR)
    # img_ceiling = cv2.resize(img_ceiling, (1280, 720))
    h, w, c = img_ceiling.shape

    img_angle = cv2.imread('./data/sample/angle2.png', cv2.IMREAD_COLOR)
    # img_angle = cv2.resize(img_angle, (1280, 720))


    c_pts1 = np.float32([[613, 18], [86, 416], [701, 14],
                         [1231, 356]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    c_pts2 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])  # 이미지 전체 크기
    c_H = cv2.getPerspectiveTransform(c_pts1, c_pts2)
    img_top_ceil = cv2.warpPerspective(img_ceiling, c_H, (1280, 720))

    a_pts1 = np.float32([[585, 21], [94, 357], [659, 27],
                         [1158, 367]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_pts2 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])  # 이미지 전체 크기
    a_H = cv2.getPerspectiveTransform(a_pts1, a_pts2)
    img_top_angle = cv2.warpPerspective(img_angle, a_H, (1280, 720))



    # c_top_pts = np.float32([[3, 414], [0, 530], [1279, 412], [1276, 522]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    # a_top_pts = np.float32([[4, 518], [3, 582], [1273, 472], [1275, 535]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    # a_top_pts = np.float32([[2, 571], [4, 518], [1266, 539], [1265, 468]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]

    c_top_pts = np.float32([[19, 416], [6, 716], [1146, 404], [1275, 708]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_top_pts = np.float32([[5, 566], [9, 142], [1269, 527], [1274, 69]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]

    ca_H = cv2.getPerspectiveTransform(c_top_pts, a_top_pts)
    ac_H = cv2.getPerspectiveTransform(a_top_pts, c_top_pts)

    # # # ##이미지 좌표 따오기
    # paste_coordinate = get_four_points(img_top_angle)
    # print("paste_coordinate: ", paste_coordinate)
    # paste_coordinate = np.array(paste_coordinate)


    # ceil_pig = [595, 32, 609, 65] #나
    # ceil_pig = [636, 24, 649, 52] #경윤
    ceil_pig = [550, 41, 563, 84] #진호

    ceil_to_topceil = transform_polygon(c_H, rectangle_to_polygon(
        ceil_pig))  # BB [626.22114081 185.16983677 742.21929358 328.59193591] #xmin, ymin, xmax, ymax
    top_ceil_to_top_angled = transform_polygon(ca_H, ceil_to_topceil)
    top_angled_to_angled = transform_polygon(np.linalg.inv(a_H), top_ceil_to_top_angled)
    print("test: ", top_angled_to_angled)
    # img_result = cv2.rectangle(img_top_angle, (35, 375), (206, 511), (0, 255, 0), 5)
    img_result = cv2.rectangle(img_angle, (-8731, 3608), (573, 45), (0, 255, 0), 5)


    cv2.imshow('test', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





