import cv2
import itertools
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


def getGridCells(img):
    cells = []
    side = img.shape[:1]
    side = side[0] / 9
    for j, i in itertools.product(range(9), range(9)):
        # Top left corner of a cell
        p1 = (i * side, j * side)
        # Bottom right corner of a cell
        p2 = ((i + 1) * side, (j + 1) * side)
        cells.append((p1, p2))
    return cells


def cutOutCell(img, cell):
    return img[int(cell[0][1]) : int(cell[1][1]), int(cell[0][0]) : int(cell[1][0])]


def getLargestFeatureInCell(img, topL=None, bottomR=None):
    img = img.copy()
    height, width = img.shape[:2]
    maxArea = 0
    seedPoint = (0, 0)

    if topL is None:
        topL = [0, 0]

    if bottomR is None:
        bottomR = [width, height]

    for x in range(topL[0], bottomR[0]):
        for y in range(topL[1], bottomR[1]):
            # Only operate on light or white squares
            # Note that .item() appears to take input as y, x
            if img.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > maxArea:
                    maxArea = area[0]
                    seedPoint = (x, y)

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)
    if all(p is not None for p in seedPoint):
        cv2.floodFill(img, mask, seedPoint, 255)

    top, bottom, left, right = height, 0, width, 0
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:
                cv2.floodFill(img, mask, (x, y), 0)

            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype="float32"), seedPoint


def scaleAndCentre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background square."""
    height, width = img.shape[:2]

    def centrePad(length):
        """Handles centering for a given length that may be odd or even."""
        side1 = int((size - length) / 2)
        side2 = side1 if length % 2 == 0 else side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if height > width:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / height
        width, height = scale(ratio, width), scale(ratio, height)
        l_pad, r_pad = centrePad(width)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / width
        width, height = scale(ratio, width), scale(ratio, height)
        t_pad, b_pad = centrePad(height)

    if width > 0:
        img = cv2.resize(img, (width, height))
    img = cv2.copyMakeBorder(
        img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background
    )
    return cv2.resize(img, (size, size))


def extractDigit(img, cell, size):
    digit = cutOutCell(img, cell)
    height, width = digit.shape[:2]
    margin = int(np.mean([height, width]) / 2.5)
    _, bbox, seed = getLargestFeatureInCell(
        digit, [margin, margin], [width - margin, height - margin]
    )
    digit = cutOutCell(digit, bbox)
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scaleAndCentre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)


def getDigitsFromCells(img, cells, size):
    img = preProcessImage(img.copy())
    return [extractDigit(img, cell, size) for cell in cells]


def showImg(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showDigits(digits, colour=255):
    """Shows list of 81 extracted digits in a grid format"""
    rows = []
    with_border = [
        cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour)
        for img in digits
    ]
    for i in range(9):
        row = np.concatenate(with_border[i * 9 : ((i + 1) * 9)], axis=1)
        rows.append(row)

    return np.concatenate(rows)


def getBoard():
    # Get the original img
    originalImg = getOriginalImage(159)
    # showImg("Original", originalImg)

    # Set Thresholds up on the img
    preProcessedImg = preProcessImage(originalImg)
    # showImg("After Thresholding", preProcessedImg)

    # Get the sudoku grid borders
    gridWithBorder, gridEdges = getGridBorder(originalImg.copy(), preProcessedImg)
    # showImg("Grid with Border", gridWithBorder)

    # Get the grid corner points
    corners = getGridCornerPoints(gridEdges)

    # get the extracted grid on its own
    croppedGridImg = cropAndWarpImg(originalImg, corners)
    # showImg("Grid Only", croppedGridImg)

    gridCells = getGridCells(croppedGridImg)
    # print(gridCells)

    digits = getDigitsFromCells(croppedGridImg, gridCells, 50)
    processedImg = showDigits(digits)
    showImg("Processed Grid", processedImg)


getBoard()
