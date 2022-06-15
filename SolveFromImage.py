import cv2
import numpy as np
import operator


def getOriginalImage(imgName):
    imgPath = f"puzzles/{imgName}.jpg"
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    return img


def preProcessImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    img = cv2.bitwise_not(img, img)
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.uint8)
    img = cv2.dilate(img, kernel)
    return img


def getGridBorder(originalImg, processedImg):
    contours, _ = cv2.findContours(
        processedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img = cv2.drawContours(originalImg, contours, 0, (0, 0, 255), 10)
    return [img, contours[0]]


def getGridCornerPoints(grid):
    # Use "Ramer Doughlas Peucker algorithm" to get corners
    bottom_right, _ = max(
        enumerate(pt[0][0] + pt[0][1] for pt in grid), key=operator.itemgetter(1)
    )
    top_left, _ = min(
        enumerate(pt[0][0] + pt[0][1] for pt in grid), key=operator.itemgetter(1)
    )
    bottom_left, _ = min(
        enumerate(pt[0][0] - pt[0][1] for pt in grid), key=operator.itemgetter(1)
    )
    top_right, _ = max(
        enumerate(pt[0][0] - pt[0][1] for pt in grid), key=operator.itemgetter(1)
    )

    return [
        grid[top_left][0],
        grid[top_right][0],
        grid[bottom_right][0],
        grid[bottom_left][0],
    ]


def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a**2) + (b**2))


def cropAndWarpImg(img, corners):
    top_left, top_right, bottom_right, bottom_left = (
        corners[0],
        corners[1],
        corners[2],
        corners[3],
    )

    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    side = max(
        [
            distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),
            distance_between(top_left, top_right),
        ]
    )
    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32"
    )
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))


def showImg(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getBoard():
    # Get the original img
    originalImg = getOriginalImage(159)
    showImg("Original", originalImg)

    # Set Thresholds up on the img
    preProcessedImg = preProcessImage(originalImg)
    showImg("After Thresholding", preProcessedImg)

    # Get the sudoku grid borders
    gridWithBorder, gridEdges = getGridBorder(originalImg.copy(), preProcessedImg)
    showImg("Grid with Border", gridWithBorder)

    # Get the grid corner points
    corners = getGridCornerPoints(gridEdges)

    # get the extracted grid on its own
    gridOnlyImg = cropAndWarpImg(originalImg, corners)
    showImg("Grid Only", gridOnlyImg)


getBoard()
