import cv2
import mediapipe as mp
import numpy as np
from cv2 import aruco

def black_hat(landmarks):
    im_path = 'black_hat.png'
    threshold = [200, 255]
    key_points = [
        (125, 487),  # 左下
        (546, 487),  # 右下
        (128, 123),  # 左上
        (583, 120)  # 右上
    ]
    dst_points = [
        (landmarks[54][0], landmarks[54][1] - 10),  # 左
        (landmarks[284][0], landmarks[284][1] - 10),  # 右
        (landmarks[103][0], landmarks[103][1] - 120),  # 下
        (landmarks[332][0], landmarks[332][1] - 120)  # 上
    ]
    return im_path, threshold, key_points, dst_points

def pink_hat(landmarks):
    im_path = 'hat_pink.png'
    threshold = [225, 255]
    key_points = [
        (138, 535),  # 左下
        (568, 535),  # 右下
        (207, 279),  # 左上
        (536, 279)  # 右上
    ]
    dst_points = [
        (landmarks[54][0], landmarks[54][1] - 30),  # 左
        (landmarks[284][0], landmarks[284][1] - 30),  # 右
        (landmarks[103][0], landmarks[103][1] - 120),  # 下
        (landmarks[332][0], landmarks[332][1] - 120)  # 上
    ]
    return im_path, threshold, key_points, dst_points

def sunglass_1(landmarks):
    im_path = 'sunglass.png'
    threshold = [150, 255]
    key_points = [
        (400, 908),
        (900, 908),
        (400, 998),
        (900, 998)
    ]
    dst_points = [
        (landmarks[124][0], landmarks[124][1]),  # 左上
        (landmarks[353][0], landmarks[353][1]),  # 右上
        (landmarks[117][0], landmarks[117][1]),  # 左下
        (landmarks[346][0], landmarks[346][1])  # 右下
    ]
    return im_path, threshold, key_points, dst_points

def sunglass_2(landmarks):
    im_path = 'SUNGLASSES_2.jpg'
    threshold = [225, 255]
    key_points = [
        (120, 110),
        (592 - 120, 110),
        (120, 180),
        (592 - 120, 180)
    ]
    dst_points = [
        (landmarks[124][0], landmarks[124][1]),  # 左上
        (landmarks[353][0], landmarks[353][1]),  # 右上
        (landmarks[117][0], landmarks[117][1]),  # 左下
        (landmarks[346][0], landmarks[346][1])  # 右下
    ]
    return im_path, threshold, key_points, dst_points

def load_item_data(item_function, landmarks):
    im_path, threshold, key_points, dst_points = item_function(landmarks)
    item_img = cv2.imread(im_path)
    gray = cv2.cvtColor(item_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold[0], threshold[1], cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(item_img)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    item_extracted = cv2.bitwise_and(item_img, mask)
    return item_extracted, key_points, dst_points

# 获取人脸特征点
def get_face_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)
    if result.multi_face_landmarks:
        landmarks = []
        for face_landmarks in result.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                landmarks.append((x, y))
        return landmarks, result
    return None, None

def warp_item(item_img, src_points, dst_points, frame):
    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32), np.array(dst_points, dtype=np.float32))
    # Apply the perspective transformation
    warped_item = cv2.warpPerspective(item_img, matrix, (frame.shape[1], frame.shape[0]))
    return warped_item

# Read Aruco Marker
def read_aruco_marker(image, dictionary):
    corner_list, id_list, _ = cv2.aruco.detectMarkers(image, dictionary)
    id = np.array([])
    if id_list is not None:
        for k in range(len(id_list)):
            id_k = id_list[k][0]
            corners_k = corner_list[k][0]
            if id.size == 0:
                id = np.array([id_k])
            else:
                id = np.concatenate((id, [id_k]), axis=0)
        return id
    return id

# 初始化Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Load Aruco Dictionary
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 选择要显示的物件
selected_hat = 0  # 选择粉帽子
selected_glasses = 0  # 选择墨镜2
display_hat = False  # Flag to determine if hat should be displayed
display_glasses = False  # Flag to determine if glasses should be displayed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, result = get_face_landmarks(frame)

    aruco_reading = read_aruco_marker(frame, dictionary)
    print(aruco_reading)

    if 5 in aruco_reading:
        display_hat = False
        display_glasses = False
        selected_hat = 0
        selected_glasses = 0
    else:
        for i in range(aruco_reading.size):
            if aruco_reading[i] == 1 or aruco_reading[i] == 2:
                selected_hat = aruco_reading[i]
                display_hat = True
            elif aruco_reading[i] == 3 or aruco_reading[i] == 4:
                selected_glasses = aruco_reading[i]
                display_glasses = True

    if landmarks:
        if display_hat:
            if selected_hat == 1:
                item_img, src_points, dst_points = load_item_data(black_hat, landmarks)
            elif selected_hat == 2:
                item_img, src_points, dst_points = load_item_data(pink_hat, landmarks)

            if selected_hat in [1, 2]:
                warped_item = warp_item(item_img, src_points, dst_points, frame)
                mask = np.any(warped_item != 0, axis=-1)
                frame[mask] = warped_item[mask]

        if display_glasses:
            if selected_glasses == 3:
                item_img, src_points, dst_points = load_item_data(sunglass_1, landmarks)
            elif selected_glasses == 4:
                item_img, src_points, dst_points = load_item_data(sunglass_2, landmarks)

            if selected_glasses in [3, 4]:
                warped_item = warp_item(item_img, src_points, dst_points, frame)
                mask = np.any(warped_item != 0, axis=-1)
                frame[mask] = warped_item[mask]

    cv2.imshow('Virtual Clothes and Sunglasses Fitting', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
