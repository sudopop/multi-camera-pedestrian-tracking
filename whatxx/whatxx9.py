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

    cv2.waitKey(8000)
    # while True:
    #     cv2.waitKey(1)
    #     if len(data) == 4:
    #         break

    points = np.array(data['points'], dtype=float)

    print(points)

    return points


if __name__ == '__main__':
    # Read source image
    img_src = cv2.imread('./data/sample/ceiling.png', cv2.IMREAD_COLOR)
    # img_src = cv2.imread('./data/sample/woman-1807533_960_720.jpg', cv2.IMREAD_COLOR)
    # img_src = cv2.resize(img_src, (1920, 1080))
    img_src = cv2.resize(img_src, (960, 540))
    h, w, c = img_src.shape

    img_dest = cv2.imread('./data/sample/angle.png', cv2.IMREAD_COLOR)
    # img_dest = cv2.imread('./data/sample/billboard-g7005ff0f9_1920.png', cv2.IMREAD_COLOR)
    # img_dest = cv2.resize(img_dest, (1920, 1080))
    img_dest = cv2.resize(img_dest, (960, 540))

    # Get source image parameter: [[left,top], [left,bottom], [right, top], [right, bottom]]
    img_src_coordinate = get_four_points(img_src)
    img_src_coordinate = np.array(img_src_coordinate)

    paste_coordinate = get_four_points(img_dest)
    paste_coordinate = np.array(paste_coordinate)

    matrix, _ = cv2.findHomography(img_src_coordinate, paste_coordinate, 0)

    #object
    # object_coordinate = get_four_points(img_src)
    # object_coordinate = np.array(object_coordinate)
    # # List to Tuple (->int) # https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=zerosum99&logNo=120193873324
    # ptr1 = [int(x) for x in object_coordinate[0]]
    # ptr1 = tuple(ptr1)
    # print("ptr1: ", ptr1)
    # ptr2 = [int(x) for x in object_coordinate[3]]
    # ptr2 = tuple(ptr2)
    # print("ptr2: ", ptr2)
    # img_src = cv2.rectangle(img_src, ptr1, ptr2, (0, 0, 255), 1)

    # object 간편 (xmin, ymin, xmax, ymax)
    img_src = cv2.rectangle(img_src, (140, 189), (187, 297), (0, 0, 255), 1)
    ceil_track = [140, 189, 187, 297]

    perspective_img = cv2.warpPerspective(img_src, matrix, (img_dest.shape[1], img_dest.shape[0]))  # 0: 1080 1:1920
    cv2.imshow('perspective_img', perspective_img)
    cv2.imshow('img_dest', img_dest)

    # cv2.copyTo(src=perspective_img, mask=np.tile(perspective_img, 1), dst=img_dest)
    # cv2.imshow('result', img_dest)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




