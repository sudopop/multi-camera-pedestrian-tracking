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
    img_ceiling = cv2.imread('./data/sample/ceiling.png', cv2.IMREAD_COLOR)
    img_ceiling = cv2.resize(img_ceiling, (960, 540))
    h, w, c = img_ceiling.shape

    img_angle = cv2.imread('./data/sample/angle.png', cv2.IMREAD_COLOR)
    img_angle = cv2.resize(img_angle, (960, 540))


    #ceiling 탑뷰 호모그래피
    # c_pts1 = np.float32([[167, 157], [71, 528], [477, 138], [483, 535]]) #돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    c_pts1 = np.float32([[217, 71], [86, 536], [476, 60], [481, 532]]) #돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    c_pts2 = np.float32([[10, 10], [10, 540], [960, 10], [960, 540]]) #이미지 전체 크기
    c_H = cv2.getPerspectiveTransform(c_pts1, c_pts2)
    img_top_ceil = cv2.warpPerspective(img_ceiling, c_H, (960, 540))

    # ##이미지 좌표 따오기
    # paste_coordinate = get_four_points(img_top_ceil)
    # print("paste_coordinate: ", paste_coordinate)
    # paste_coordinate = np.array(paste_coordinate)

    # angle 탑뷰 호모그래피
    a_pts1 = np.float32([[384, 14], [163, 449], [582, 5], [805, 449]])  #돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_pts2 = np.float32([[10, 10], [10, 540], [960, 10], [960, 540]]) #이미지 전체 크기
    a_H = cv2.getPerspectiveTransform(a_pts1, a_pts2)
    img_top_angle = cv2.warpPerspective(img_angle, a_H, (960, 540))

    # angle 탑뷰 -> ceil 탑뷰 호모그래피 (겹치는 부분)
    # a_top_pts = np.float32([[5, 0], [5, 300], [948, 13], [951, 289]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    # c_top_pts = np.float32([[6, 4], [7, 531], [954, 4], [956, 532]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    a_top_pts = np.float32([[5, 0], [5, 300], [948, 13], [951, 289]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    c_top_pts = np.float32([[7, 128], [9, 535], [948, 129], [952, 527]])  # 돼지우리 땅바닥 좌표 [[left,top], [left,bottom], [right, top], [right, bottom]]
    ac_H = cv2.getPerspectiveTransform(a_top_pts, c_top_pts) #역함수에 사용할 경우 H (결과는 같음)
    ca_H = cv2.getPerspectiveTransform(c_top_pts, a_top_pts) #그냥 사용할 경우 H (결과는 같음)

    # 테스트: object 간편하게 수기입력 (xmin, ymin, xmax, ymax)
    ceil_pig = [140, 189, 187, 297]
    ceil_to_topceil = transform_polygon(c_H, rectangle_to_polygon(ceil_pig))  # BB [626.22114081 185.16983677 742.21929358 328.59193591] #xmin, ymin, xmax, ymax
    # top_ceil_to_top_angled = transform_polygon(np.linalg.inv(ac_H), ceil_to_topceil)  # 역함수 H로 top_angle -> top_ceil로 변환하는데 사용한 값
    top_ceil_to_top_angled = transform_polygon(ca_H, ceil_to_topceil)  # 역함수 대신 그냥 H로 ceil top-> angle top
    top_angled_to_angled = transform_polygon(np.linalg.inv(a_H), top_ceil_to_top_angled)
    print("test: ", top_angled_to_angled)
    img_result = cv2.rectangle(img_angle, (366, 21), (405, 52), (0, 255, 0), 5)


    # img_result = cv2.rectangle(img_ceiling, (140, 189), (187, 297), (0, 255, 0), 2)

    cv2.imshow('test', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #
    ## 객체 정상적으로 적용되는지 테스트





    # img_ceiling = cv2.rectangle(img_ceiling, (140, 189), (187, 297), (0, 0, 255), 1)
    #
    #
    # test = cv2.warpPerspective(img_top_angle, ac_H, (960, 540))
    # inv_test = transform_polygon(np.linalg.inv(ac_H), test)



    # #ceil 객체 -> ceil 탑뷰에 적용 테스트
    # ceil_to_topceil = transform_polygon(c_H, rectangle_to_polygon(ceil_pig))
    # print("ceil_to_topceil: ", ceil_to_topceil) #{'tl': [-50, 67], 'tr': [100, 71], 'br': [173, 248], 'bl': [37, 246]}
    # img_top_ceil_obj = cv2.rectangle(img_top_ceil, (-50, 67), (173, 248), (255, 0, 0), 10)
    #
    # # ceil 탑뷰 -> ceil로 적용 테스트(역함수)
    # topceil_to_ceil = transform_polygon(np.linalg.inv(c_H), ceil_to_topceil)
    # print("topceil_to_ceil: ", topceil_to_ceil)  # {'tl': [140, 188], 'tr': [186, 188], 'br': [186, 296], 'bl': [140, 296]}
    # img_ceiling = cv2.rectangle(img_ceiling, (140, 188), (186, 296), (255, 0, 0), 10)

    # perspective_img = cv2.warpPerspective(img_src, M, (img_dest.shape[1], img_dest.shape[0]))  # 0: 1080 1:1920
    # cv2.imshow('perspective_img', perspective_img)

    # matrix, _ = cv2.findHomography(img_src_coordinate, paste_coordinate, 0)

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

    # # object 간편 (xmin, ymin, xmax, ymax)
    # img_ceiling = cv2.rectangle(img_ceiling, (140, 189), (187, 297), (0, 0, 255), 1)
    # ceil_pig = [140, 189, 187, 297]
    #
    # perspective_img = cv2.warpPerspective(img_ceiling, matrix, (img_angle.shape[1], img_angle.shape[0]))  # 0: 1080 1:1920
    # cv2.imshow('perspective_img', perspective_img)
    # cv2.imshow('img_dest', img_angle)
    #
    # # cv2.copyTo(src=perspective_img, mask=np.tile(perspective_img, 1), dst=img_dest)
    # # cv2.imshow('result', img_dest)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




