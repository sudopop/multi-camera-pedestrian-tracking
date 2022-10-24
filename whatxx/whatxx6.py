##출처: https://icodebroker.tistory.com/5251 [ICODEBROKER:티스토리]

import matplotlib
import numpy as np
import skimage.color as color
from skimage.color import rgb2gray
import skimage.feature as feature
import skimage.io as io
import skimage.measure as measure
import skimage.transform as transform
from skimage.transform import resize

matplotlib.use("TkAgg")

io.use_plugin("matplotlib")

# imageNDArray1 = io.imread("./data/sample/source1.png")
# imageNDArray2 = io.imread("./data/sample/source2.png")

imageNDArray1 = io.imread("./data/sample/angle.png")
imageNDArray1 = resize(imageNDArray1, (960, 720))
imageNDArray1 = rgb2gray(imageNDArray1)
imageNDArray2 = io.imread("./data/sample/ceiling.png")
imageNDArray2 = resize(imageNDArray2, (960, 720))
imageNDArray2 = rgb2gray(imageNDArray2)

orb = feature.ORB(n_keypoints = 1000, fast_threshold = 0.05)

orb.detect_and_extract(imageNDArray1)

keypointNDArray1   = orb.keypoints
descriptorNDArray1 = orb.descriptors

orb.detect_and_extract(imageNDArray2)

keypointNDArray2   = orb.keypoints
descriptorNDArray2 = orb.descriptors

matchNDArray = feature.match_descriptors(descriptorNDArray1, descriptorNDArray2, cross_check = True)

sourceNDArray = keypointNDArray2[matchNDArray[:, 1]][:, ::-1]
targetNDArray = keypointNDArray1[matchNDArray[:, 0]][:, ::-1]

projectiveTransform, inlierNDArray = measure.ransac((sourceNDArray, targetNDArray), transform.ProjectiveTransform, min_samples = 4, residual_threshold = 2)

y, x = imageNDArray2.shape[:2]

cornerNDArray       = np.array([[0, 0], [0, y], [x, 0],[x, y]])
warpedCornerNDArray = projectiveTransform(cornerNDArray)
allCornerNDArray    = np.vstack((warpedCornerNDArray, cornerNDArray))

minimumCornerNDArray = np.min(allCornerNDArray, axis = 0)
maximumCornerNDArray = np.max(allCornerNDArray, axis = 0)

outputShapeNDArray = (maximumCornerNDArray - minimumCornerNDArray)
outputShapeNDArray = np.ceil(outputShapeNDArray[::-1])

similarityTransform = transform.SimilarityTransform(translation = -minimumCornerNDArray)

warpImageNDArray1 = transform.warp(imageNDArray1, similarityTransform.inverse, output_shape = outputShapeNDArray, cval = -1)
warpImageNDArray2 = transform.warp(imageNDArray2, (projectiveTransform + similarityTransform).inverse, output_shape = outputShapeNDArray, cval = -1)

maskImageNDArray1 = (warpImageNDArray1 != -1)

warpImageNDArray1[~maskImageNDArray1] = 0

alphaImageNDArray1 = np.dstack((color.gray2rgb(warpImageNDArray1), maskImageNDArray1))

maskImageNDArray2 = (warpImageNDArray2 != -1)

warpImageNDArray2[~maskImageNDArray2] = 0

alphaImageNDArray2 = np.dstack((color.gray2rgb(warpImageNDArray2), maskImageNDArray2))

mergeImageNDArray = (alphaImageNDArray1 + alphaImageNDArray2)

alphaImageNDArray = mergeImageNDArray[..., 3]

mergeImageNDArray /= np.maximum(alphaImageNDArray, 1)[..., np.newaxis]

io.imshow(mergeImageNDArray)

io.show()
