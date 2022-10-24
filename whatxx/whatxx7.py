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
    cv2.waitKey(5000)

    points = np.array(data['points'], dtype=float)

    print(points)

    return points


if __name__ == '__main__':
    # Read source image
    img_src = cv2.imread('./data/sample/woman-1807533_960_720.jpg', cv2.IMREAD_COLOR)
    # img_src = cv2.resize(img_src, (960, 720))
    # cv2.imwrite('source_image.jpg', img_src)
    h, w, c = img_src.shape
    # Get source image parameter: [[left,top], [left,bottom], [right, top], [right, bottom]]
    img_src_coordinate = np.array([[0,0],[0,h],[w,0],[w,h]])
    # Read destination image
    img_dest = cv2.imread('./data/sample/billboard-g7005ff0f9_1920.png', cv2.IMREAD_COLOR)
    img_dest = cv2.resize(img_dest, (1920, 1080))
    #1920*1080
    # copy destination image for get_paste_position (Just avoid destination image will be draw)
    img_dest_copy = img_dest.copy()#np.tile(img_dest, 1)
    # paste_coordinate in destination image
    # paste_coordinate = []
    # cv2.namedWindow('collect coordinate')
    # cv2.setMouseCallback('collect coordinate', get_paste_position, paste_coordinate)
    # while True:
    #     cv2.waitKey(1)
    #     if len(paste_coordinate) == 4:
    #         break

    paste_coordinate = get_four_points(img_dest)

    paste_coordinate = np.array(paste_coordinate)
    # Get perspective matrix
    matrix, _ = cv2.findHomography(img_src_coordinate, paste_coordinate, 0)

    #국진
    corners = [10,10,30,30]
    img_src = cv2.rectangle(img_src, (10,10), (30,30), (0,0,255), 10)

    ceil_to_topceil = transform_polygon(matrix, rectangle_to_polygon(corners))
    print(ceil_to_topceil)

    print(f'matrix: {matrix}')
    perspective_img = cv2.warpPerspective(img_src, matrix, (img_dest.shape[1], img_dest.shape[0])) #0: 1080 1:1920
    cv2.imshow('img', perspective_img)
    cv2.copyTo(src=perspective_img, mask=np.tile(perspective_img, 1), dst=img_dest)
    cv2.imshow('result', img_dest)
    cv2.waitKey()
    cv2.destroyAllWindows()

    pts1 = np.array([[844, 274], [861, 272], [861, 285], [844, 286]], dtype=np.int32)

    cv2.polylines(img_dest, [pts1], False, (0, 0, 255), 10)
    cv2.imshow('polyline', img_dest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()