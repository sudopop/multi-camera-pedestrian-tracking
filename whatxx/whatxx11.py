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
    # img_ceiling = cv2.resize(img_ceiling, (960, 540))
    h, w, c = img_ceiling.shape

    img_angle = cv2.imread('./data/sample/angle2.png', cv2.IMREAD_COLOR)
    # img_angle = cv2.resize(img_angle, (960, 540))



    c_pts1 = np.float32([[612, 17], [89, 420], [697, 17],
                         [1231, 356]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    c_pts2 = np.float32([[0, 0], [0, 720], [640, 10], [640, 720]])  # 이미지 전체 크기
    c_H = cv2.getPerspectiveTransform(c_pts1, c_pts2)
    img_top_ceil = cv2.warpPerspective(img_ceiling, c_H, (640, 720))

    a_pts1 = np.float32([[581, 22], [90, 362], [661, 23],
                         [1135, 365]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_pts2 = np.float32([[0, 0], [0, 720], [640, 10], [640, 720]])  # 이미지 전체 크기
    a_H = cv2.getPerspectiveTransform(a_pts1, a_pts2)
    img_top_angle = cv2.warpPerspective(img_angle, a_H, (640, 720))

    # # ##이미지 좌표 따오기
    paste_coordinate = get_four_points(img_ceiling)
    print("paste_coordinate: ", paste_coordinate)
    paste_coordinate = np.array(paste_coordinate)

    c_top_pts = np.float32([[49, 438], [36, 536], [589, 389],
                            [579, 514]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_top_pts = np.float32([[70, 477], [44, 534], [566, 479],
                            [584, 536]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    # a_top_pts = np.float32([[46, 533], [69, 480], [588, 540],
    #                         [565, 481]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    ca_H = cv2.getPerspectiveTransform(c_top_pts, a_top_pts)



    # ceil_pig = [595, 30, 608, 68] #나
    # ceil_pig = [636, 24, 649, 52] #흰티
    ceil_pig = [550, 41, 563, 84] #진호
    ceil_to_topceil = transform_polygon(c_H, rectangle_to_polygon(
        ceil_pig))  # BB [626.22114081 185.16983677 742.21929358 328.59193591] #xmin, ymin, xmax, ymax
    top_ceil_to_top_angled = transform_polygon(ca_H, ceil_to_topceil)
    # top_angled_to_angled = transform_polygon(np.linalg.inv(a_H), top_ceil_to_top_angled)
    print("test: ", top_ceil_to_top_angled)
    # img_result = cv2.rectangle(img_top_angle, (35, 375), (206, 511), (0, 255, 0), 5)
    img_result = cv2.rectangle(img_top_angle, (-131, 414), (107, 536), (0, 255, 0), 5)
    # # img_result = cv2.rectangle(img_ceiling, (595, 30), (608, 68), (0, 255, 0), 5)
    #
    #
    #
    cv2.imshow('test', img_top_angle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #
    # c_pts1 = np.float32([[565, 56], [550, 81], [730, 43],
    #                      [758, 73]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    # c_pts2 = np.float32([[517, 81], [538, 66], [725, 87], [705, 67]])
    # c_H = cv2.getPerspectiveTransform(c_pts1, c_pts2)
    # img_top_ceil = cv2.warpPerspective(img_ceiling, c_H, (1280, 720))
    #
    # ceil_pig = [565, 56, 758, 73]
    # ceil_to_topceil = transform_polygon(c_H, rectangle_to_polygon(
    #     ceil_pig))  # BB [626.22114081 185.16983677 742.21929358 328.59193591] #xmin, ymin, xmax, ymax
    # print("test: ", ceil_to_topceil)
    # img_result = cv2.rectangle(img_angle, (517, 81), (705, 66), (0, 255, 0), 5)

    #
    # cv2.imshow('test', img_top_ceil)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()





