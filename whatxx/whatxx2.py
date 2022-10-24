import cv2
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

img_src = cv2.imread('./data/sample/ceiling.png')
img_src = cv2.resize(img_src, (400, 300))

start_point = (140, 140)
end_point = (200, 200)
color = (0, 0, 255)
thickness = 3
image_with_rectangle = cv2.rectangle(
    img = img_src,
    pt1 = start_point,
    pt2 = end_point,
    color = color,
    thickness = thickness
)

# 결과물 이미지 사이즈
dst_size = (400, 300, 3)

# 결과의 이미지를 넣을 행렬 , 0행렬
img_dst = np.zeros(dst_size, np.uint8)  # 빈 행렬만들기. 0

# cv2.imshow('dst', img_dst)


# 우리가 원본이미지로부터는 마우스 클릭으로 4개의 점을 가져온다.
# 새로만들 이미지에서는, 위의 원본 이미지 4개의 점과 매핑할 점을 잡아줘야한다.

# cv2.imshow('image',img_src)

# 함수를 호출하여 이미지 작업한다.
points_src = get_four_points(img_src)  # 함수호출

print("points_src")

points_dst = np.array([0, 0, dst_size[1], 0,
                       dst_size[1], dst_size[0], 0, dst_size[0]], dtype=float)

points_dst = points_dst.reshape(4, 2)

h, status = cv2.findHomography(points_src, points_dst)
M = cv2.getPerspectiveTransform(pts1, pts2)

img_dst = cv2.warpPerspective(img_src, h, (dst_size[1], dst_size[0]))


cv2.imshow('result', img_dst)

cv2.waitKey()
cv2.destroyAllWindows()


# ceil_to_topceil = transform_polygon(h, rectangle_to_polygon([140, 140, 200, 200]))
#
# print(ceil_to_topceil)
# # pts = np.array([[10,5], [20,30], [70,20], [50,10]], np.int32)
# pts = []
# for pos, old_p in ceil_to_topceil.items():
#     pts.append(old_p)
# print(pts)
#
# n_pts=np.array(pts)
# # pts = pts.reshape((-1, 1, 2))
# img_src = cv2.polylines(img_src, [n_pts], True, (0,255,255))
#
# cv2.imshow('img_src', img_src)