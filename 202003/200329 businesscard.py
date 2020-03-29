# 명함인식 구현하기 - 웹캠 (2)

import numpy as np
import cv2


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered corrdinates
    return rect


def auto_scan_image_via_webcam():
    try:
        cap = cv2.VideoCapture(0)
    except:
        print('cannot load camera!')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print('cannot load camera!')
            break

        k = cv2.waitKey(10)
        if k == 27:
            break

        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(gray, 75, 200)

        # show the original image and the dege detected image
        print("STEP 1: Edge Detection")

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        conts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            screenCnt = []

            # if our approximated contour has four points, the we
            # can assume that we have found our screen
            if len(approx) == 4:
                contourSize = cv2.contourArea(approx)
                camSize = frame.shape[0] * frame.shape[1]
                ratio = contourSize / camSize
                print(contourSize)
                print(camSize)
                print(ratio)

                if ratio > 0.1:
                    screenCnt = approx

                break

            if len(screenCnt) == 0:
                cv2.imshow("WebCam", frame)
                continue
            else:
                # show the contour (outline) of the piece of paper
                print("STEP 2: FInd contours of paper")

                cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
                cv2.imshow("WebCam", frame)

                # apply the four point transform to obtain a top-down
                # view of the original image
                rect = order_points(screenCnt, reshape(4, 2))
                (topLeft, topRIght, bottomRight, bottomLeft) = rect

                w1 = abs(bottomRight[0] - bottomLeft[0])
                w2 = abs(topRIght[0] - topLeft[0])
                h1 = abs(topRight[1] - bottomRight[1])
                h2 = abs(topLeft[1] - bottomLeft[1])
                maxWidth = max([w1, w2])
                maxHeight = max([h1, h2])

                dst = np.float32([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]])

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(frame, M, (maxWIdth, maxHeight))

                # show the original and scanned images
                print("STEP 3: Apply perspective transform")

                # convert the warped image to grayscale, then threshold it
                # to give it that 'black and white' paper effect
                warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.tHRESH_BINARY, 21, 10)

                # show the original and scanned images
                print("STEP 4: Apply Adaptive Threshold")

                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    cv2.imshow("Scanned", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    auto_scan_image_via_webcam()