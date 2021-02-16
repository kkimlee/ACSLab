from ctypes import *
import random
import os
import cv2
import time
import barcode_darknet
import argparse
import numpy as np
from threading import Thread, enumerate
from queue import Queue

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from enum import Enum
import barcode as bc

# 3bit
class ColorSet(Enum):
    C000 = [255,0,0]
    C001 = [255,191.5,0]
    C011 = [127.5,255,0]
    C010 = [0,255,63.5]
    C110 = [0,255,255]
    C111 = [0,63.5,255]
    C101 = [127.5,0,255]
    C100 = [255,0,191.5]

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video

def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)

def video_capture(frame_queue, darknet_image_queue):

    while cap.isOpened():
        print('video_capture')
        ret, frame = cap.read()
        # 이미지 회전
        # frame = cv2.transpose(frame)
        # frame = cv2.flip(frame, 0)
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame_resized)
        img_for_detect = barcode_darknet.make_image(width, height, 3)
        barcode_darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        print('inference')
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = barcode_darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        
        barcode_darknet.print_detections(detections, args.ext_output)
        barcode_darknet.free_image(darknet_image)
        
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (width, height))
    
    w = 350
    h = 10
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('barcode.avi', fourcc, fps, (w, h))
    
    #bit_list
    bit_dict = {
        "000": ColorSet.C000.value,
        "001": ColorSet.C001.value,
        "011": ColorSet.C011.value,
        "010": ColorSet.C010.value,
        "110": ColorSet.C110.value,
        "111": ColorSet.C111.value,
        "101": ColorSet.C101.value,
        "100": ColorSet.C100.value
    }

    origin_bit_list = np.array(list(bit_dict.keys()))

    # 3bit
    origin_data_list = [1, 2, 3, 4, 5, 6, 7] * 3
    origin_data_list = origin_data_list + [0, 1, 2, 3, 4, 5]
    np.array(origin_data_list)

    rgb_channel_matrix = [[0,0,0], [0,0,0], [0,0,0]]
    cmy_channel_matrix = [[0,0,0], [0,0,0], [0,0,0]]

    rx_data_list = list()
    rx_data_list2 = list()
    rx_data_list3 = list()
    rx_data_list4 = list()
    rx_data_list5 = list()

    num_frame = 0
    sync = 0

    ber = list()
    ber2 = list()
    ber3 = list()
    ber4 = list()
    ber5 = list()

    r_pos = 0
    g_pos = 0
    b_pos = 0
    
    # 바코드 프레임 저장하기 위한 인덱스
    idx = 0
    while cap.isOpened():
        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        
        if frame_resized is not None:
            # 바코드 슬라이싱
            barcode = barcode_darknet.barcode(detections, frame_resized)
            barcode = cv2.cvtColor(barcode, cv2.COLOR_BGR2RGB)
            barcode = cv2.resize(barcode, (350, 10))
            
            image = barcode_darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if args.out_filename is not None:
                video.write(image)
            if not args.dont_show:
                cv2.imshow('Inference', image)
                
                if idx < 10:
                    strIdx = "00"+str(idx)
                elif idx < 100:
                    strIdx = "0"+str(idx)
                else:
                    strIdx = str(idx)
                    
                cv2.imshow('Barcode', barcode)
                # print(input_path[:-4])
                cv2.imwrite(input_path[:-4]+'/'+strIdx+'.jpg', barcode)
                out.write(barcode)
                
                idx += 1
                
            rgb_start, rgb_end, rgb_max, rgb_min, rgb_pilot, pos = bc.GetPixelGraph(barcode, 0, 349, 0)
            if sync == 0:
                if rgb_pilot == [1, 1, 0, 0, 0, 0]:
                    sync += 1
                    r_pos = pos[0]
                    r, g, b, c, m, y= bc.GetColor(barcode, 0, 349, rgb_start, rgb_end, pos)
                    rgb_channel_matrix[0] = [r, g, b]
                    cmy_channel_matrix[0] = [c, m, y]
            elif sync == 1:
                if rgb_pilot == [0, 0, 1, 1, 0, 0]:
                    sync += 1
                    g_pos = pos[1]
                    r, g, b, c, m, y= bc.GetColor(barcode, 0, 349, rgb_start, rgb_end, pos)
                    rgb_channel_matrix[1] = [r, g, b]
                    cmy_channel_matrix[1] = [c, m, y]
                else:
                    sync = 0
                    r_pos = 0
                    g_pos	 = 0
                    b_pos = 0
                    rgb_channel_matrix = [[0,0,0], [0,0,0], [0,0,0]]
                    cmy_channel_matrix = [[0,0,0], [0,0,0], [0,0,0]]
            elif sync == 2:
                if rgb_pilot == [0, 0, 0, 0, 1, 1]:
                    sync += 1
                    b_pos = pos[2]
                    r, g, b, c, m, y= bc.GetColor(barcode, 0, 349, rgb_start, rgb_end, pos)
                    rgb_channel_matrix[2] = [r, g, b]
                    cmy_channel_matrix[2] = [c, m, y]
                    print("sync")
            
                else:
                    sync = 0
                    r_pos = 0
                    g_pos = 0
                    b_pos = 0
                    rgb_channel_matrix = [[0,0,0], [0,0,0], [0,0,0]]
                    cmy_channel_matrix = [[0,0,0], [0,0,0], [0,0,0]]
            elif sync == 3:
                rgb_pos = [r_pos, g_pos, b_pos]
                r, g, b, c, m, y = bc.GetColor(barcode, 0, 349, rgb_start, rgb_end, pos)
                rx_color = bc.estimating(r, g, b, rgb_channel_matrix)
                rx_data = bc.decoding(rx_color)
                rx_data_list.append(rx_data)
        
                rx_color2 = bc.estimating(c, m, y, cmy_channel_matrix)
                rx_data2 = bc.decoding(rx_color2)
                rx_data_list2.append(rx_data2)
        
                rx_color3 = (rx_color + rx_color2)/2
                rx_data3 = bc.decoding(rx_color3)
                rx_data_list3.append(rx_data3)
        
                rx_color4 = bc.estimating(r, g, b, cmy_channel_matrix)
                rx_data4 = bc.decoding(rx_color4)
                rx_data_list4.append(rx_data4)
        
                rx_data5 = bc.decoding2(rx_color, rx_color2)
                rx_data_list5.append(rx_data5)
        
        
        
                num_frame += 1
        
            if num_frame == 27:
                sync = 0
                num_frame = 0
                r_pos = 0
                g_pos = 0
                b_pos = 0
                rgb_channel_matrix = [[0,0,0], [0,0,0], [0,0,0]]
                cmy_channel_matrix = [[0,0,0], [0,0,0], [0,0,0]]
        
                ber.append(bc.calculateBER(origin_data_list, rx_data_list, origin_bit_list))
                ber2.append(bc.calculateBER(origin_data_list, rx_data_list2, origin_bit_list))
                ber3.append(bc.calculateBER(origin_data_list, rx_data_list3, origin_bit_list))
                ber4.append(bc.calculateBER(origin_data_list, rx_data_list4, origin_bit_list))
                ber5.append(bc.calculateBER(origin_data_list, rx_data_list5, origin_bit_list))
                rx_data_list = list()
                rx_data_list2 = list()
                rx_data_list3 = list()
                rx_data_list4 = list()
                rx_data_list5 = list()
                print('sync out')  
                
            if cv2.waitKey(fps) == 27:
                print('finish1')
                break
    print(ber)
    print(ber2)
    print(ber3)
    print(ber4)
    print(ber5)                
    print('finish2')
    out.release() 
    cap.release()
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    barcode_queue = Queue()
    fps_queue = Queue(maxsize=1)
    
    fps = 0
    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = barcode_darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    width = barcode_darknet.network_width(network)
    height = barcode_darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path) 
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()
