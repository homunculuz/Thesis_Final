import cv2 as cv
aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
board = cv.aruco.CharucoBoard_create(16, 9, 1, .8, aruco_dict)
arucoParams = cv.aruco.DetectorParameters_create()

imboard = board.draw((2000, 2000))
cv.imwrite("chessboard.jpg", imboard)


