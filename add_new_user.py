#!/usr/bin/env python3
import cv2
from multiprocessing import Process
import multiprocessing
import time
from time import sleep
import os
from datetime import datetime, timedelta
import sys
import numpy as np
import math


def transform_frame(temp_image, empty_frame):
    pre_frame = empty_frame
    angle = 110
    (he, we, c) = empty_frame.shape
    empty_frame_yc = int(he / 2)
    empty_frame_xc = int(we / 2)
    (h, w, c) = temp_image.shape
    half_h = int(h / 2)
    half_w = int(w / 2)
    pre_frame[
    empty_frame_yc - half_h:
    empty_frame_yc + half_h,
    empty_frame_xc - half_w:
    empty_frame_xc + half_w, :
    ] = temp_image
    transformation_matrix = cv2.getRotationMatrix2D((empty_frame_xc, empty_frame_yc), angle, 1)
    transformed_image = cv2.warpAffine(pre_frame, transformation_matrix, (we, he))
    return transformed_image


def cam_thread(stop):
    import logging
    logger = logging.getLogger('camera_thread')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log_add_user_camera_thread.txt')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    capture_pipeline = (
        'v4l2src device=/dev/video0 ! '
        'image/jpeg, width=1280, height=720, framerate=30/1, format=MJPG ! '
        'jpegparse ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink'
    )
    cam = cv2.VideoCapture(capture_pipeline, cv2.CAP_GSTREAMER)

    mask1 = cv2.imread('mask1_2.jpg', 0)
    Logo = cv2.imread('logo2.jpg')
    logger.info('Loaded media')

    final_w = 1680
    final_h = 1050
    final_frame_cam_frame_x0 = 0
    final_frame_delta_minus = 86
    black_frame = np.zeros((final_h, final_h, 3), np.uint8)
    final_frame = np.zeros((final_h, final_w, 3), np.uint8)
    aspect_x = 16
    aspect_y = 10
    diag = int(math.sqrt(aspect_x ** 2 + aspect_y ** 2))
    step = int(final_h / diag) - 1
    cam_resize_x = step * aspect_x
    cam_resize_y = step * aspect_y

    time.sleep(10)
    os.system('xrandr -o normal')
    os.system('v4l2-ctl -d 0 -c zoom_absolute=160')
    os.system('v4l2-ctl -d 0 -c pan_absolute=0')
    os.system('v4l2-ctl -d 0 -c tilt_absolute=0')
    os.system('v4l2-ctl -d 0 -c brightness=140')
    logger.info('camera config done')

    window_name = 'Video'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    skip_frames_MAX = 6
    skip_frames = 0

    t0 = time.time()
    total_faces = 0
    first = True

    while True:
        try:
            ret, frame = cam.read()
            if not ret:
                continue

            now_time = time.time()

            if (now_time - t0) < 2:
                frame = cv2.imread('timer3.jpg')
            elif (now_time - t0) < 4:
                frame = cv2.imread('timer2.jpg')
            elif (now_time - t0) < 6:
                frame = cv2.imread('timer1.jpg')
            elif (now_time - t0) > 20:
                stop.set()
                break
            else:
                if first:
                    cv2.imwrite('NewUser/face_ID.jpg', frame)
                    first = False
                else:
                    skip_frames = skip_frames + 1
                    if skip_frames >= skip_frames_MAX:
                        cv2.imwrite('NewUser/{}.jpg'.format(total_faces), frame)
                        skip_frames = 0
                        total_faces = total_faces + 1

            rot_frame = cv2.resize(frame, (cam_resize_x, cam_resize_y))
            rot_frame = cv2.flip(rot_frame, 1)
            rot_frame = transform_frame(rot_frame, black_frame)
            final_frame[
            0:final_h,
            final_frame_cam_frame_x0:final_frame_cam_frame_x0 + final_h - final_frame_delta_minus
            ] = rot_frame[:, final_frame_delta_minus:]

            final_frame = cv2.bitwise_and(final_frame, final_frame, mask=mask1)
            final_frame = cv2.bitwise_or(final_frame, Logo)
            final_frame = cv2.flip(final_frame, -1)
            cv2.imshow(window_name, final_frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                stop.set()
                break
            if stop.is_set():
                break
        except Exception as e:
            logger.error('exception in while loop')
    cam.release()
    logger.info('done!')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # os.system('echo 190 > /sys/devices/pwm-fan/target_pwm')
    multiprocessing.set_start_method('forkserver')
    global_stop = multiprocessing.Event()
    global_stop.clear()

    camera_process = Process(target=cam_thread,
                             args=(global_stop,), daemon=True)
    camera_process.start()

    while True:
        try:
            if global_stop.is_set():
                break
            sleep(1)
        except KeyboardInterrupt:
            global_stop.set()
    camera_process.terminate()
