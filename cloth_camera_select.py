import cv2
import numpy as np
from cv2 import aruco
from screeninfo import get_monitors
import time
import mediapipe as mp

monitor = get_monitors()[1]
a_size_x = monitor.height
a_size_y = monitor.width


BLUE = (255, 0, 0)
GREEN = (0, 0, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)

rec_x0 = 220
rec_y0 =  190
size_x = 200
size_y = 100
rect_pos = np.array([[rec_x0,rec_y0],[rec_x0+size_x,rec_y0],[rec_x0+size_x,rec_y0+size_y],[rec_x0,rec_y0+size_y]])
rect_pos = rect_pos.astype(np.int64)

clothes_img_path = np.array(['content/male.jpg','content/cloth2.jpg','content/cloth3.jpg','content/cloth4.png','content/cloth5.jpg','content/cloth6.jpg'])
clothes_img_path_1 = np.array(['content/cloth1.png','content/cloth2.bmp','content/cloth3.png','content/cloth4.jpg','content/cloth5.jpg','content/cloth6.png'])

cloth_landamark = np.array([[(161, 275), (404, 275), (207, 572), (330, 579)], #cloth1
                            [(258, 109), (702,109), (300, 803), (600,803)], #cloth2
                            [(136, 222), (273, 231), (141, 430), (204, 435)], #cloth3
                            [(388,488),  (895,487), (471, 1200),(808,1200)],#cloth4
                            [(130,74),   (380,74),  (142, 474), (367,474)], #cloth5
                            [(173, 202), (333, 202),(193, 391), (293, 391)],#cloth6
                            ])

selection_threshold = 3.0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def warp_clothes(clothes_img, landmarks, frame, cloth_landmarks):
    # 定义衣服图片上的源点（source points）
    src_points = np.array(cloth_landmarks, dtype=np.float32)

    # 定义人体上的目标点（destination points）
    dst_points = np.array([
        landmarks[11],  # 左肩
        landmarks[12],  # 右肩
        landmarks[23],  # 左肘
        landmarks[24]  # 右肘
    ], dtype=np.float32)

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # 进行透视变换
    warped_clothes = cv2.warpPerspective(clothes_img, matrix, (frame.shape[1], frame.shape[0]))

    return warped_clothes

def add_aruco(canvas, point_list):

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    side_length = 360

    aruco_count = 8

    gap = (a_size_y - side_length - 60) / (aruco_count/2 - 1)

    aruco_position = np.array([[],[]])

    if point_list.size != 0:
        if point_list[0][0] >= 0 & point_list[0][1] >= 0:
            cv2.circle(canvas, point_list[0], 50, BLUE, cv2.FILLED)
        # cv2.polylines(canvas, [point_list],True,0,10)

    for marker_id in range(int(aruco_count/2)):
        # aruco.drawMarker() generates a monochrome image
        marker_image_top = aruco.drawMarker(dictionary, int(marker_id * 2), side_length)

        marker_image_top = cv2.cvtColor(marker_image_top, cv2.COLOR_GRAY2BGR)


        # print(marker_image_top.shape)

        # That is why the canvas image also must be monochrome
        x0 = int(marker_id * gap + 30)
        y0 = 30

        aruco_bg = np.full((side_length + y0 * 2, side_length + y0 * 2, 3), 255, dtype=np.uint8)

        canvas[y0 - 30 :y0+side_length + 30, x0-30:x0+side_length+30] = aruco_bg

        canvas[y0:y0+side_length, x0:x0+side_length] = marker_image_top


        aruco_pos = np.array([[x0 + side_length/2, y0 + side_length/2]])
        
        if aruco_position.size == 0:
            aruco_position = aruco_pos
        else:
            aruco_position = np.concatenate((aruco_position,aruco_pos),axis = 0)
        


        marker_image_bottom = aruco.drawMarker(dictionary, marker_id * 2 + 1 , side_length)

        x1 = int(marker_id * gap + 30)
        y1 = int(a_size_x - side_length - 30)

        marker_image_bottom = cv2.cvtColor(marker_image_bottom, cv2.COLOR_GRAY2BGR)

        canvas[y1 - 30 :y1+side_length + 30, x1-30:x1+side_length+30] = aruco_bg
     
        canvas[y1:y1+side_length, x1:x1+side_length] = marker_image_bottom

        aruco_pos = np.array([[x1 + side_length/2, y1 + side_length/2]])

        aruco_position = np.concatenate((aruco_position,aruco_pos),axis = 0)

    # print(canvas.shape)

    

    cv2.namedWindow('canvas', cv2.WND_PROP_FULLSCREEN)
    # Move the window to that monitor
    cv2.moveWindow('canvas', monitor.x, monitor.y)
    # Remove the titlebar and so on
    cv2.setWindowProperty('canvas',
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    cv2.imshow('canvas', canvas)

    return  aruco_position

def selections(point_list,stage):

    canvas = np.full((a_size_x, a_size_y, 3), 255, dtype=np.uint8)

    # canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    if stage == 0:

        y = int(a_size_y / (len(clothes_img_path) / 2))
        x = int(a_size_x / 2)

        for i in range(len(clothes_img_path)):
            cloth = clothes_img_path[i]
            cloth_img = cv2.imread(cloth)
            cloth_img = cv2.resize(cloth_img,(y,x))
            x0 = int(np.remainder(i,2) * x)
            y0 = int(np.floor(i/2) * y)

            # print(x)
            # print(y)
            # print(cloth_img.shape)
            # print(a_size_x)
            # print(x0)
            # print(y0)
            canvas[ x0:x0 + x, y0: y0 + y ]  = cloth_img
        
        if point_list.size != 0:
            if point_list[0][0] >= 0 & point_list[0][1] >= 0:
                select = np.floor(point_list[0][1] / x) + np.floor(point_list[0][0] / y) * 2
                select = int(select)
                return canvas, select

    # if stage == 0:
    #     male = clothes_img_path[0]
    #     male_img = cv2.imread(male)
    #     male_img = cv2.resize(male_img,(1920,2160))
    #     female = 'content/male.jpg'
    #     female_img = cv2.imread(female)
    #     male_image_height, male_image_width = male_img.shape[:2]
    #     canvas[0:2160, 0:1920] = male_img
    #     print(canvas.shape)

    return canvas, None

def get_body_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)
    if result.pose_landmarks:
        landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in result.pose_landmarks.landmark]
        return landmarks, result
    return None, None

def main():
    
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    cap = cv2.VideoCapture(2)
    cap2 = cv2.VideoCapture(4)

    # cv2.namedWindow('camera preview')
    
    successful, image = cap.read()


    # rect_pos_ref = np.concatenate((rect_pos.transpose(),np.array([[1,1,1,1]])),axis = 0)

    rect_pos_ref = [[int(image.shape[1]/2)],[int(image.shape[0]/2)],[1]]

    # print(rect_pos_ref)

    rect_project = np.array([])

    previous_selection = -1

    canvas,_ = selections(rect_project,0)

    aruco_position = add_aruco(canvas, rect_project)

    select_time0 = None

    y = int(a_size_y / (len(clothes_img_path) / 2))
    x = int(a_size_x / 2)

    box_corner = np.full((len(clothes_img_path), 2, 2), 0, dtype=np.int32)

    for i in range(len(clothes_img_path)):
        box_corner[i][0][0] = int(np.remainder(i,2) * x)
        box_corner[i][0][1] = int(np.floor(i/2) * y)
        box_corner[i][1][0] = int(box_corner[i][0][0] + x)
        box_corner[i][1][1] = int(box_corner[i][0][1] + y)


    while True:
        successful, image = cap.read()

        if not successful:
            break

        # Detect markers and obtain the lists of marker corners and marker IDs
        # screen_size = image.shape
        # size_x = 200
        # size_y = 100
        # rec_x0 = int(np.round(screen_size[1]/2-size_x/2,decimals=0))
        # rec_y0 = int(np.round(screen_size[0]/2-size_y/2,decimals=0))
        # rec_x1 = int(np.round(screen_size[1]/2+size_x/2,decimals=0))
        # rec_y1 = int(np.round(screen_size[0]/2+size_y/2,decimals=0))

        # print(rec_x0, rec_y0)
        # print(screen_size)
        # print(rec_x0)

        #                    top-left, bottom-right, color, thickness
        # cv2.rectangle(image, (rec_x0, rec_y0), (rec_x1, rec_y1), GREEN, 2)

        corner_list, id_list, _ = cv2.aruco.detectMarkers(image, dictionary)
        current_time = time.time()

        # pts_src = np.array()
        # pts_dst = np.array()


        aruco_position_camera = np.array([[],[]])
        aruco_position_camera_cores = np.array([[],[]])

        if id_list is not None:
            for k in range(len(id_list)):
                id_k = id_list[k][0]
                corners_k = corner_list[k][0]
                # print('ID = ', id_k)
                # print('corners = ', corners_k)
                
                center = np.sum(corners_k,axis=0)/4
                center = center.round(decimals=0)

                if id_k < 8:
                
                    if aruco_position_camera.size == 0:
                        aruco_position_camera = np.array([center])
                    else:
                        aruco_position_camera = np.concatenate((aruco_position_camera,np.array([center])),axis = 0)
                    
                    if aruco_position_camera_cores.size == 0:
                        aruco_position_camera_cores = np.array([aruco_position[id_k]])
                    else:
                        aruco_position_camera_cores = np.concatenate((aruco_position_camera_cores,np.array([aruco_position[id_k]])),axis = 0)


                cv2.circle(image, np.int32(center), 5, RED, cv2.FILLED)
                # cv2.circle(image, np.int32(corners_k[0]), 3, BLUE, cv2.FILLED)
                # cv2.circle(image, np.int32(corners_k[1]), 3, GREEN, cv2.FILLED)
                # cv2.circle(image, np.int32(corners_k[2]), 3, RED, cv2.FILLED)
                # cv2.circle(image, np.int32(corners_k[3]), 3, CYAN, cv2.FILLED)
                cv2.putText(image, 'ID = ' + str(id_k),
                            np.int32(corners_k[0]), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=RED)

        print(flush=True)

        aruco_position_camera_cores = aruco_position_camera_cores.astype(int)
        aruco_position_camera = aruco_position_camera.astype(int)

        # print(aruco_position_camera_cores)

        # print(aruco_position_camera_cores.shape)

        if aruco_position_camera_cores.shape[0] >= 4 :
            h, status = cv2.findHomography(aruco_position_camera,aruco_position_camera_cores)
            # print(status)
            if status[0] != 0:
                rect_project = np.matmul(h,rect_pos_ref)
                # rect_project = np.transpose(np.array([rect_project[0]],[rect_project[1]])).astype(np.int32)
                rect_project = np.concatenate(([rect_project[0]],[rect_project[1]]),axis = 0).astype(np.int32)
                rect_project = rect_project.transpose()
                print(rect_project)
        else:
            status = False
            
        canvas,select = selections(rect_project, 0)

        print(select)

        if select != None and select <= len(clothes_img_path) - 1:
            if select != previous_selection:
                previous_selection = select
                select_time0 = current_time
            elif select_time0 != None:
                if current_time - select_time0 > selection_threshold:
                    break
                else:
                    cv2.rectangle(canvas, (box_corner[select][0][1], box_corner[select][0][0]), (box_corner[select][1][1], box_corner[select][1][0]), GREEN, 20)
                    progress_percentage = int(box_corner[select][0][1] + y * (current_time - select_time0) / selection_threshold)
                    left_top_x = int(box_corner[select][0][0] + x *0.9)
                    cv2.rectangle(canvas, (box_corner[select][0][1], left_top_x), (progress_percentage, box_corner[select][1][0]), GREEN, cv2.FILLED)

        # cv2.polylines(image, [rect_pos], True, GREEN, 2)
        
        # cv2.imshow('camera preview', image)

        aruco_position = add_aruco(canvas, rect_project)

        # print(aruco_position)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()


    while True:
        successful, image = cap2.read()

        if not successful:
            print('camera failed')
            break

        selected_clothes_img_path = clothes_img_path_1[select]
        clothes_img = cv2.imread(selected_clothes_img_path )
        gray = cv2.cvtColor(clothes_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(clothes_img)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        clothes_extracted = cv2.bitwise_and(clothes_img, mask)
        # clothes_extracted_rgb = cv2.cvtColor(clothes_extracted, cv2.COLOR_BGR2RGB)
        selected_cloth_landmark = cloth_landamark[select]
        extracted = True

        landmarks, result = get_body_landmarks(image)

        if landmarks and extracted:

            warped_clothes = warp_clothes(clothes_extracted, landmarks, image, selected_cloth_landmark)
            mask = np.any(warped_clothes != 0, axis=-1)
            image[mask] = warped_clothes[mask]
            # mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.namedWindow('Fit Room', cv2.WND_PROP_FULLSCREEN)
        # Move the window to that monitor
        cv2.moveWindow('Fit Room', monitor.x, monitor.y)
        # Remove the titlebar and so on
        cv2.setWindowProperty('Fit Room',
                            cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Fit Room', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
