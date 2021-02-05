import cv2
import colorsys # rgb to hsl
import os
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
from enum import Enum
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.signal import argrelextrema
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
from scipy import signal

frames_dict = {
    "barcode": 0,
    "X`R": 1,
    "RR`": 2,
    "R`G": 3,
    "GG`": 4,
    "G`B": 5,
    "BB`": 6,
    "B`X": 7
}

# RGB
# openCV는 BGR임을 유의
class Color(Enum):
    RED = [255, 0, 0]
    REDCOM = [0, 255, 255]
    GREEN = [0, 255, 0]
    GREENCOM = [255, 0, 255]
    BLUE = [0, 0, 255]
    BLUECOM = [255, 255, 0]

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

# # origin_bit_list = np.array(bit_dict.keys())
# class ColorSet(Enum):
#     C0000 = [255,0,0]
#     C0001 = [255,95,0]
#     C0011 = [255,191,0]
#     C0010 = [223,255,0]
#     C0110 = [127,255,0]
#     C0111 = [31,255,0]
#     C0101 = [0,255,63]
#     C0100 = [0,255,159]
#     C1100 = [0,255,255]
#     C1101 = [0,159,255]
#     C1111 = [0,63,255]
#     C1110 = [31,0,255]
#     C1010 = [127,0,255]
#     C1011 = [223,0,255]
#     C1001 = [255,0,191]
#     C1000 = [255,0,95]
#
# # bit_list
# bit_dict = {
#     "0000": ColorSet.C0000.value,
#     "0001": ColorSet.C0001.value,
#     "0011": ColorSet.C0011.value,
#     "0010": ColorSet.C0010.value,
#     "0110": ColorSet.C0110.value,
#     "0111": ColorSet.C0111.value,
#     "0101": ColorSet.C0101.value,
#     "0100": ColorSet.C0100.value,
#     "1100": ColorSet.C1100.value,
#     "1101": ColorSet.C1101.value,
#     "1111": ColorSet.C1111.value,
#     "1110": ColorSet.C1110.value,
#     "1010": ColorSet.C1010.value,
#     "1011": ColorSet.C1011.value,
#     "1001": ColorSet.C1001.value,
#     "1000": ColorSet.C1000.value
# }

origin_bit_list = np.array(list(bit_dict.keys()))

# # 4bit
# origin_data_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
# origin_data_list = origin_data_list + [ 11, 12, 13, 14, 15 ,0, 1]
# np.array(origin_data_list)

# 3bit
origin_data_list = [1, 2, 3, 4, 5, 6, 7] * 3
origin_data_list = origin_data_list + [0, 1, 2, 3, 4, 5]
np.array(origin_data_list)

def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)


def contours_processing(img, margin, y_start, y_end):
    print("y_start, y_end")
    print(y_start, y_end)
    for idx in range(y_start, y_end + 1):
        print(idx)
        x_list = np.argwhere(img[idx, :] > 0)
        print(x_list)

        if x_list.size > 0:
            x_start = x_list.min()
            x_end = x_list.max()
            print(x_start, x_end)
            img[idx, x_start:x_start + margin + 1] = 0
            img[idx, x_end - margin:x_end + 1] = 0

    return img

def getHue(r_mean, g_mean, b_mean, pos, near=10):
    h, l, s = colorsys.rgb_to_hls(r_mean[pos], g_mean[pos], b_mean[pos])
    return h

# channel matrix 구하는 부분
# RGB Pilot Frame
def getChannelMatrix(rx_matrix):
    origin_matrix = [Color.RED.value, Color.GREEN.value, Color.BLUE.value]
    print(rx_matrix[0])

    channel_matrix = np.dot(rx_matrix, np.linalg.inv(origin_matrix))
    print(channel_matrix)

    return channel_matrix

def estimating(r, g, b, channel_matrix, pseudo=False):
    color = np.array([[r], [g], [b]])

    print("Rx")
    print(color)

    # dot H
    # print(channel_matrix)
    if pseudo == False:
        color = np.dot(np.linalg.inv(channel_matrix), color)
    else:
        print(np.linalg.pinv(channel_matrix))
        color = np.dot(np.linalg.pinv(channel_matrix), color)
    # color = np.clip(color, 0 , 255)

    print("Est")
    print(color)

    return color

def decoding(color):
    est_h, est_l, est_s = colorsys.rgb_to_hls(color[0],color[1],color[2])

    # print(color)
    # print(est_h,est_s,est_l)

    code_hls_list = []
    for item in ColorSet:
        h, l, s = colorsys.rgb_to_hls(item.value[0], item.value[1], item.value[2])
        code_hls_list.append(h)

    code_hls_list = np.array(code_hls_list)
    print(code_hls_list)

    # decoding, maximum likelihood
    min = 1.0
    minIdx = -1
    idx = 0
    for item in code_hls_list:
        diff1 = abs(item - est_h)
        diff2 = abs(1-diff1)
        diff = np.where(diff1<diff2, diff1, diff2)
        if diff < min:
            min = diff
            minIdx = idx
        idx += 1

    print("Est:{0}, Candidate:{1}".format(est_h, code_hls_list[minIdx]))
    return minIdx

def getMid(a, b):
    return int((a+b)/2)

def calculateBER(origin_data_list, rx_data_list):
    fe = 0
    be = 0

    print(origin_data_list)
    print(rx_data_list)

    # FER, BER
    for origin_frame, rx_frame in zip(origin_data_list, rx_data_list):
        if origin_frame != rx_frame:
            fe += 1
            origin_bit = origin_bit_list[origin_frame]
            rx_bit = origin_bit_list[rx_frame]

            for origin, rx in zip(origin_bit, rx_bit):
                if origin!=rx:
                    be += 1
    return fe, be

# FFT based signal smoothing
def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = signal.convolve(w / w.sum(), s, mode='valid')
    return y


def GetPixelGraph(img, y_start, y_end, frame_type):
    blue, green, red = cv2.split(img)
    b_list = []
    g_list = []
    r_list = []
    for idx in range(y_start, y_end):
        x_list = np.argwhere(blue[idx, :])
        if x_list.size > 0:
            x_start = x_list.min()
            x_end = x_list.max()
            b_list.append(blue[idx, x_start:x_end + 1].mean())
            g_list.append(green[idx, x_start:x_end + 1].mean())
            r_list.append(red[idx, x_start:x_end + 1].mean())

    r_mean = np.array(r_list)
    g_mean = np.array(g_list)
    b_mean = np.array(b_list)

    # B, G, R 성분 분화
    rotated_b, rotated_g, rotated_r = cv2.split(img)

    # 90도 회전
    rotated_plt = cv2.merge([rotated_r.T, rotated_g.T, rotated_b.T])

    # 상하반전
    rotated_plt = cv2.flip(rotated_plt, 0)

    # Filtering Method
    # smoothing : Savitzky-Golay filter
    # r_mean_filtered = savgol_filter(r_mean, 301, 2)
    # g_mean_filtered = savgol_filter(g_mean, 301, 2)
    # b_mean_filtered = savgol_filter(b_mean, 301, 2)

    #medfilter
    # r_mean_filtered = medfilt(r_mean, 301)
    # g_mean_filtered = medfilt(g_mean, 301)
    # b_mean_filtered = medfilt(b_mean, 301)

    # New Filter - ButterWorth
    N = 3  # Filter order
    Wn = 0.005  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba', analog=False)
    r_mean_filtered = signal.filtfilt(B, A, r_mean, padtype='constant', padlen=150)
    g_mean_filtered = signal.filtfilt(B, A, g_mean, padtype='constant', padlen=150)
    b_mean_filtered = signal.filtfilt(B, A, b_mean, padtype='constant', padlen=150)

    # w, h = signal.freqs(B, A)
    # plt.plot(w, 20 * np.log10(abs(h)))
    # plt.xscale('log')
    # plt.title('Butterworth filter frequency response')
    # plt.xlabel('Frequency [radians / second]')
    # plt.ylabel('Amplitude [dB]')
    # plt.margins(0, 0.1)
    # plt.grid(which='both', axis='both')
    # plt.axvline(100, color='green') # cutoff frequency
    # plt.show()

    # Finding Extrema
    # r_emax = argrelextrema(r_mean_filtered, np.greater_equal)[0]
    # r_emin = argrelextrema(r_mean_filtered, np.less_equal)[0]
    # g_emax = argrelextrema(g_mean_filtered, np.greater_equal)[0]
    # g_emin = argrelextrema(g_mean_filtered, np.less_equal)[0]
    # b_emax = argrelextrema(b_mean_filtered, np.greater_equal)[0]
    # b_emin = argrelextrema(b_mean_filtered, np.less_equal)[0]
    r_emax = []
    r_emin = []
    g_emax = []
    g_emin = []
    b_emax = []
    b_emin = []

    # 최대 최소 고려
    r_max = np.argmax(r_mean_filtered)
    r_min = np.argmin(r_mean_filtered)
    g_max = np.argmax(g_mean_filtered)
    g_min = np.argmin(g_mean_filtered)
    b_max = np.argmax(b_mean_filtered)
    b_min = np.argmin(b_mean_filtered)

    '''
    # R, G, B graph
    plt.subplot(411), plt.imshow(rotated_plt)
    plt.subplot(412), plt.plot(r_mean, color='r'), plt.title(
        str(r_mean[r_max]) + ' ' + str(r_max) + ' ' + str(r_mean[r_min]) + ' ' + str(r_min))
    plt.axvline(x=r_max, color='r', linestyle='--')
    plt.axvline(x=r_min, color='b', linestyle='--')
    for x_ in r_emax:
        plt.axvline(x=x_, color='g', linewidth=1)
    for x_ in r_emin:
        plt.axvline(x=x_, color='k', linewidth=1)
    plt.plot(r_mean_filtered)
    plt.subplot(413), plt.plot(g_mean, color='g'), plt.title(
        str(g_mean[g_max]) + ' ' + str(g_max) + ' ' + str(g_mean[g_min]) + ' ' + str(g_min))
    plt.axvline(x=g_max, color='r', linestyle='--')
    plt.axvline(x=g_min, color='b', linestyle='--')
    for x_ in g_emax:
        plt.axvline(x=x_, color='g', linewidth=1)
    for x_ in g_emin:
        plt.axvline(x=x_, color='k', linewidth=1)
    plt.plot(g_mean_filtered)
    plt.subplot(414), plt.plot(b_mean, color='b'), plt.title(
        str(b_mean[b_max]) + ' ' + str(b_max) + ' ' + str(b_mean[b_min]) + ' ' + str(b_min))
    plt.axvline(x=b_max, color='r', linestyle='--')
    plt.axvline(x=b_min, color='b', linestyle='--')
    for x_ in b_emax:
        plt.axvline(x=x_, color='g', linewidth=1)
    for x_ in b_emin:
        plt.axvline(x=x_, color='k', linewidth=1)
    plt.plot(b_mean_filtered)
    plt.tight_layout()
    plt.show()
    '''

    pilot_red = []
    pilot_green = []
    pilot_blue = []

    rgb_emax = [r_emax, g_emax, b_emax]
    rgb_emin = [r_emin, g_emin, b_emin]
    rgb_max = [r_max, g_max, b_max]
    rgb_min = [r_min, g_min, b_min]

    print(rgb_emax)
    print(type(rgb_emax[0]))

    return rgb_emax, rgb_emin, rgb_max, rgb_min

def GetColor(img, y_start, y_end, pos, near=10):
    blue, green, red = cv2.split(img)
    b_list = []
    g_list = []
    r_list = []
    for idx in range(y_start, y_end + 1):
        x_list = np.argwhere(blue[idx, :])
        if x_list.size > 0:
            x_start = x_list.min()
            x_end = x_list.max()
            b_list.append(blue[idx, x_start:x_end + 1].mean())
            g_list.append(green[idx, x_start:x_end + 1].mean())
            r_list.append(red[idx, x_start:x_end + 1].mean())

    pos_start = 0
    pos_end = 0
    if pos+near >= y_end:
        pos_end = y_end
    else:
        pos_end = pos + near
    if pos-near <= y_start:
        pos_start = y_start
        pos_end = y_start + near
    else:
        pos_start = pos - near

    print("[{0}, {1}]".format(pos_start, pos_end))
    r_mean = np.array(r_list[pos_start:pos_end]).mean()
    g_mean = np.array(g_list[pos_start:pos_end]).mean()
    b_mean = np.array(b_list[pos_start:pos_end]).mean()

    return r_mean, g_mean, b_mean

def GetGraph(img, y_start, y_end, pos, id):
    blue, green, red = cv2.split(img)
    b_list = []
    g_list = []
    r_list = []
    for idx in range(y_start, y_end + 1):
        x_list = np.argwhere(blue[idx, :])
        if x_list.size > 0:
            x_start = x_list.min()
            x_end = x_list.max()
            b_list.append(blue[idx, x_start:x_end + 1].mean())
            g_list.append(green[idx, x_start:x_end + 1].mean())
            r_list.append(red[idx, x_start:x_end + 1].mean())

    r_mean = np.array(r_list)
    g_mean = np.array(g_list)
    b_mean = np.array(b_list)

    # B, G, R 성분 분화
    rotated_b, rotated_g, rotated_r = cv2.split(img)

    # 90도 회전
    rotated_plt = cv2.merge([rotated_r.T, rotated_g.T, rotated_b.T])

    # 상하반전
    rotated_plt = cv2.flip(rotated_plt, 0)

    # 최대 최소 고려
    r_max = np.argmax(r_mean)
    r_min = np.argmin(r_mean)
    g_max = np.argmax(g_mean)
    g_min = np.argmin(g_mean)
    b_max = np.argmax(b_mean)
    b_min = np.argmin(b_mean)

    plt.subplot(411), plt.imshow(rotated_plt)
    plt.subplot(412), plt.plot(r_mean, color='r'), plt.title(
        str(r_mean[r_max]) + ' ' + str(r_max) + ' ' + str(r_mean[r_min]) + ' ' + str(r_min))
    plt.axvline(x=pos, color='k', linestyle='--')
    plt.subplot(413), plt.plot(g_mean, color='g'), plt.title(
        str(g_mean[g_max]) + ' ' + str(g_max) + ' ' + str(g_mean[g_min]) + ' ' + str(g_min))
    plt.axvline(x=pos, color='k', linestyle='--')
    plt.subplot(414), plt.plot(b_mean, color='b'), plt.title(
        str(b_mean[b_max]) + ' ' + str(b_max) + ' ' + str(b_mean[b_min]) + ' ' + str(b_min))
    plt.axvline(x=pos, color='k', linestyle='--')
    plt.tight_layout()
    # if id >= 100:
    #     plt.savefig('./figure_analysis/' + str(id) + '.png', dpi=300)
    # elif id >= 10:
    #     plt.savefig('./figure_analysis/0' + str(id) + '.png', dpi=300)
    # else:
    #     plt.savefig('./figure_analysis/00' + str(id) + '.png', dpi=300)
    # plt.close()
    plt.show()


if __name__ == '__main__':
    video_path = "yaw_-20_res/yaw_-20_1.mp4"
    txt_path = "yaw_-20_res/yaw_-20_1.txt"

    f = open(txt_path, 'r')
    lines = f.readlines()
    line_idx = 0

    # cam 호출
    winName = "Movement Indicator"
    cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

    cap = cv2.VideoCapture(video_path)

    prev_frame = -1 # -1 : None Sync, 0 : Sync(DataFrame), etc : Find Sync
    flag_sync = False
    frame_cnt = 0
    data_size = 27

    frames_list = []

    # 최대 Bounding Box 영역 지정
    txt_length = len(lines)
    x_list = []
    y_list = []
    h_list = []
    w_list = []
    label_list = []

    # Pixel Graph 출력
    isPrintGraph = False
    isPrintDataGraph = False

    for idx in range(0, txt_length, 2):
        label = lines[idx].strip()
        coord = lines[idx+1].strip()

        l_list = label.split(", ")
        coord_list = coord.split("\t")
        print(l_list)
        print(coord_list)

        label_list.append(l_list)

        xcoord = coord_list[0]
        ycoord = coord_list[1]
        height = coord_list[2]
        width = coord_list[3]

        x_list.append(xcoord)
        y_list.append(ycoord)
        h_list.append(height)
        w_list.append(width)

    x_list = np.array(x_list).astype(float)
    y_list = np.array(y_list).astype(float)
    h_list = np.array(h_list).astype(float)
    w_list = np.array(w_list).astype(float)

    box_x = int(x_list.min())
    box_y = int(y_list.min())
    box_height = int(h_list.max()+0.5)
    box_width = int(w_list.max()+0.5)

    print(label_list)
    #######

    #### Video Cutting 진행
    #### Differential도 여기서 진행
    # Read three images first:
    cam = cv2.VideoCapture(video_path)

    winName = "Movement Indicator"
    cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

    t_minus = cv2.cvtColor(cam.read()[1][box_y:box_y + box_width, box_x:box_x + box_height], cv2.COLOR_BGR2GRAY)
    t = cv2.cvtColor(cam.read()[1][box_y:box_y + box_width, box_x:box_x + box_height], cv2.COLOR_BGR2GRAY)
    t_plus = cv2.cvtColor(cam.read()[1][box_y:box_y + box_width, box_x:box_x + box_height], cv2.COLOR_BGR2GRAY)
    diff = cv2.threshold(diffImg(t_minus, t, t_plus), 20, 255, cv2.THRESH_BINARY)[1]
    sum = diff
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((11,11), np.uint8)

    # Differential Summation
    while True:
        diff = cv2.threshold(diffImg(t_minus, t, t_plus), 20, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow(winName, diff)
        sum = sum + diff

        # Read next image
        t_minus = t
        t = t_plus
        ret, t_plus = cam.read()

        if ret == True:
            t_plus = cv2.cvtColor(t_plus[box_y:box_y + box_width, box_x:box_x + box_height], cv2.COLOR_BGR2GRAY)
        else:
            break

        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyWindow(winName)
            break

    # [SAVE] Differential Summation Result
    cv2.imshow(winName, sum)
    cv2.imwrite('Box_Diff.png',sum)
    cv2.waitKey(0)

    # [SAVE] Get Box + Differential + Morphology Transform + Threshold
    sum = cv2.morphologyEx(sum, cv2.MORPH_CLOSE, kernel)
    cv2.imshow(winName, sum)
    cv2.waitKey(0)
    sum = cv2.erode(sum, kernel2, 1)
    cv2.imshow(winName, sum)
    cv2.waitKey(0)
    sum = cv2.dilate(sum, kernel2, 1)
    cv2.imshow(winName, sum)
    cv2.waitKey(0)
    sum = cv2.threshold(cv2.morphologyEx(sum, cv2.MORPH_OPEN, kernel), 1, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow(winName, sum)
    cv2.imwrite('afterMorph.png',sum)
    sum_array = sum

    # Get Barcode range
    x_mean = sum_array.mean(axis=0)[:]
    print(x_mean.shape)
    y_mean = sum_array.mean(axis=1)[:]
    print(y_mean.shape)

    x_list = np.argwhere(x_mean!=0)
    y_list = np.argwhere(y_mean!=0)

    x_left_top = x_list.min()
    y_left_top = y_list.min()

    x_right_bottom = x_list.max()
    y_right_bottom = y_list.max()

    print(x_list, y_list)
    print(x_left_top, y_left_top)
    print(x_right_bottom, y_right_bottom)

    cv2.imshow(winName, sum_array)
    cv2.waitKey(0)

    print(sum_array[y_left_top:y_right_bottom, x_left_top:x_right_bottom])

    # 외곽의 margin 줘서 노이즈 제거
    margin_size = 1
    sum_array = contours_processing(sum_array, margin_size, y_left_top, y_right_bottom)
    y_left_top += margin_size * 2
    y_right_bottom -= margin_size * 2

    # pixel 수 계산
    pixel_cnt = np.argwhere(sum_array>0).size
    print(pixel_cnt)

    # Make 3 Channel Image
    sum = cv2.merge([sum_array, sum_array, sum_array])

    idx = 0

    # frame type 1, 2
    frame_type = 0

    # pilot_list, pilot candidate
    pilot_list = []

    # Received Pilot Barcode
    pilot_barcode = [[], [], []]

    # Received Pilot
    rx_pilot = [[0,0,0], [0,0,0], [0,0,0]]
    rx_pilot_6 = [[0,0,0], [0,0,0], [0,0,0]]

    # Channel Matrix
    channel_matrix = [[0,0,0], [0,0,0], [0,0,0]]
    channel_matrix_6 = [[0,0,0], [0,0,0], [0,0,0]]

    # Origin 6 Matrix
    origin_matrix_6 = []

    # Received Data
    rx_data_list = []

    # Data+Pilot Position
    pos = 0

    # Bit Error List
    be_list = []

    # Frame Error List
    fe_list = []

    # packet count
    packet_cnt = 0

    print("Label Size: ", len(label_list))

    # 1st, 2nd 프레임 버림
    # TODO: 1st = missing frame, 2nd = delayed frame, YOLOv3 Debug 필요
    ret, frame = cap.read()
    ret, frame = cap.read()


    ###################################
    ## Deocding
    ###################################
    while True:
        ret, frame = cap.read()
        if ret == True:
            boxed_image = frame[box_y:box_y + box_width, box_x:box_x + box_height]

            # TODO Diff On/Off 분기 필요
            # Differential 적용
            and_image = cv2.bitwise_and(boxed_image, sum)

            # Sync가 맞은 경우
            if flag_sync==True:
                print("Receiving Data Frames")

                r, g, b = GetColor(and_image, y_left_top, y_right_bottom, pos)

                frame_cnt += 1
                idx += 1

                if isPrintDataGraph == True:
                    GetGraph(and_image, y_left_top, y_right_bottom, pos_cand, idx)

                if frame_cnt < 21 and frame_cnt % 7 == 6:
                    if packet_cnt == 0:
                        print(r,g,b)
                        rx_color = estimating(r, g, b, channel_matrix)
                        rx_data = decoding(rx_color)
                        rx_data_list.append(rx_data)

                        rx_pilot_6[int((frame_cnt+1)/7)-1] = [r, g, b]
                        print(rx_pilot_6)

                        if int(frame_cnt/6) == 3:
                            origin_matrix_6.append([127.5,0,255])
                            origin_matrix_6.append([127.5,0,255])
                            origin_matrix_6.append([127.5,0,255])
                            origin_matrix_6 = np.array(origin_matrix_6).T
                            rx_pilot_6 = np.array(rx_pilot_6).T
                            print(rx_pilot_6, origin_matrix_6)
                            channel_matrix_6 = np.dot(rx_pilot_6, np.linalg.pinv(origin_matrix_6))
                            print(channel_matrix_6)
                    else:
                        print(frame_cnt)
                        print(channel_matrix_6)
                        rx_color = estimating(r, g, b, channel_matrix_6, True)
                        rx_data = decoding(rx_color)
                        rx_data_list.append(rx_data)
                else:
                    rx_color = estimating(r, g, b, channel_matrix)
                    rx_data = decoding(rx_color)
                    rx_data_list.append(rx_data)

                if frame_cnt == data_size:
                    pos = 0
                    frame_cnt = 0
                    flag_sync = False
                    prev_frame = -1
                    pilot_list = []
                    rx_pilot = [[0,0,0], [0,0,0], [0,0,0]]
                    channel_matrix = [[0,0,0], [0,0,0], [0,0,0]]
                    pilot_list.clear()

                    fe, be = calculateBER(origin_data_list, rx_data_list)
                    rx_data_list.clear()
                    print("FE: {0}, BE: {1}".format(fe, be))
                    be_list.append(be)
                    fe_list.append(fe)
                    packet_cnt+=1

            else:
                frames_list = []

                # Missing Frame 처리 부분
                print("FRAME No.",idx)
                if idx >= len(label_list):
                    break;

                print(label_list[idx])
                for label in label_list[idx]:
                    print(label)
                    frames_list.append(frames_dict[label])

                print("FRAME LIST: ", frames_list)
                # TODO: Barcode Labelling이 2개 이상인 경우에 대한 처리가 필요함
                # TODO: Pilot Position에 대한 확실한 결정이 필요
                if len(frames_list) > 1:
                    # Preprocessing
                    cv2.imshow(winName, and_image)
                    cv2.imwrite('barcode.png', and_image)
                    rgb_emax, rgb_emin, rgb_max, rgb_min = GetPixelGraph(and_image, y_left_top, y_right_bottom,
                                                                         frame_type)
                    r_emax, r_max = rgb_emax[0], rgb_max[0]
                    r_emin, r_min = rgb_emin[0], rgb_min[0]
                    g_emax, g_max = rgb_emax[1], rgb_max[1]
                    g_emin, g_min = rgb_emin[1], rgb_min[1]
                    b_emax, b_min = rgb_emax[2], rgb_max[2]
                    b_emin, b_min = rgb_emin[2], rgb_min[2]

                    # 이전 프레임까지 Sync 맞추기를 하는 상황
                    if prev_frame == -1 and frames_list[1] <= 2:
                        print("FIRST FRAMES_LIST[1:] : ",frames_list[1:])
                        for frame in frames_list[1:]:
                            print("Prev {0}, Cur {1}".format(prev_frame, frame))
                            if frame==1 or frame==2:
                                # R 추출
                                prev_frame = frame
                                print("[RED]Start - Synchronizing: ", frame)

                                if len(r_emax):
                                    r_max = int((np.array(r_emax).mean() + r_max) / 2)
                                if len(g_emin):
                                    g_min = int((np.array(g_emin).mean() + g_min) / 2)
                                if len(b_emin):
                                    b_min = int((np.array(b_emin).mean() + b_min) / 2)

                                #pos_cand = int((r_max + g_min + b_min) / 3)
                                #pos_cand = int((r_max+g_min)/2)
                                #pos_cand = int((g_min+b_min)/2)
                                pos_cand = g_min
                                pilot_list.append(pos_cand)
                                pilot_barcode[0] = and_image
                                rx_pilot[0] = GetColor(and_image, y_left_top, y_right_bottom, pos_cand)
                                if isPrintGraph == True:
                                    GetGraph(and_image, y_left_top, y_right_bottom, pos_cand, idx)
                            elif ((prev_frame==1 or prev_frame==2) and (frame==3 or frame==4)) or (prev_frame==3 and frame==4):
                                # G가 같이 있는 경우
                                prev_frame = frame
                                print("[GREEN]Synchronizing: ", frame)

                                r_emin, r_min = rgb_emin[0], rgb_min[0]
                                g_emax, g_max = rgb_emax[1], rgb_max[1]
                                b_emin, b_min = rgb_emin[2], rgb_min[2]

                                if len(r_emin):
                                    r_min = int((np.array(r_emin).mean() + r_min) / 2)
                                if len(g_emax):
                                    g_max = int((np.array(g_emax).mean() + g_max) / 2)
                                if len(b_emin):
                                    b_min = int((np.array(b_emin).mean() + b_min) / 2)

                                #pos_cand = int((r_min + g_max + b_min) / 3)
                                #pos_cand = int((g_max+b_min)/2)
                                #pos_cand = int((r_min+b_min)/2)
                                pos_cand = b_min
                                pilot_list.append(pos_cand)
                                pilot_barcode[1] = and_image
                                rx_pilot[1] = GetColor(and_image, y_left_top, y_right_bottom, pos_cand)
                                if isPrintGraph == True:
                                    GetGraph(and_image, y_left_top, y_right_bottom, pos_cand, idx)
                            else:
                                print("Missing Frame: ", frame)
                                pos = 0
                                frame_cnt = 0
                                flag_sync = False
                                prev_frame = -1
                                pilot_list = []
                                rx_pilot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                                pilot_barcode = [[], [], []]
                                channel_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                                pilot_list.clear()
                    else:
                        print("SECOND FRAMES_LIST[1:] : ",frames_list[1:])
                        for frame in frames_list[1:]:
                            print("Prev {0}, Cur {1}".format(prev_frame, frame))
                            if prev_frame==1 and frame==2:
                                # R이 또 다시 나온 경우
                                prev_frame = frame
                                print("[RED]Start - Synchronizing: ", frame)

                                if len(r_emax):
                                    r_max = int((np.array(r_emax).mean() + r_max) / 2)
                                if len(g_emin):
                                    g_min = int((np.array(g_emin).mean() + g_min) / 2)
                                if len(b_emin):
                                    b_min = int((np.array(b_emin).mean() + b_min) / 2)

                                #pos_cand = int((r_max + g_min + b_min) / 3)
                                #pos_cand = int((r_max+g_min)/2)
                                #pos_cand = int((g_min+b_min)/2)
                                pos_cand = g_min
                                pilot_list.append(pos_cand)
                                pilot_barcode[0] = and_image
                                rx_pilot[0] = GetColor(and_image, y_left_top, y_right_bottom, pos_cand)
                                if isPrintGraph == True:
                                    GetGraph(and_image, y_left_top, y_right_bottom, pos_cand, idx)

                                print("R: ", rx_pilot[0])
                            elif ((prev_frame==1 or prev_frame==2) and (frame==3 or frame==4)) or (prev_frame==3 and frame==4):
                                # G가 같이 있는 경우
                                prev_frame = frame
                                print("[GREEN]Synchronizing: ", frame)

                                r_emin, r_min = rgb_emin[0], rgb_min[0]
                                g_emax, g_max = rgb_emax[1], rgb_max[1]
                                b_emin, b_min = rgb_emin[2], rgb_min[2]

                                if len(r_emin):
                                    r_min = int((np.array(r_emin).mean() + r_min) / 2)
                                if len(g_emax):
                                    g_max = int((np.array(g_emax).mean() + g_max) / 2)
                                if len(b_emin):
                                    b_min = int((np.array(b_emin).mean() + b_min) / 2)

                                #pos_cand = int((r_min + g_max + b_min) / 3)
                                #pos_cand = int((g_max+b_min)/2)
                                #pos_cand = int((r_min+b_min)/2)
                                pos_cand = b_min
                                pilot_list.append(pos_cand)
                                pilot_barcode[1] = and_image
                                rx_pilot[1] = GetColor(and_image, y_left_top, y_right_bottom, pos_cand)
                                if isPrintGraph == True:
                                    GetGraph(and_image, y_left_top, y_right_bottom, pos_cand, idx)
                            elif ((prev_frame==3 or prev_frame==4) and (frame==5 or frame==6)) or (prev_frame==5 and frame==6):
                                prev_frame = frame
                                print("[BLUE]Synchronizing: ", frame)

                                r_emin, r_min = rgb_emin[0], rgb_min[0]
                                g_emin, g_min = rgb_emin[1], rgb_min[1]
                                b_emax, b_max = rgb_emax[2], rgb_max[2]

                                if len(r_emin):
                                    r_min = int((np.array(r_emin).mean() + r_min) / 2)
                                if len(g_emin):
                                    g_min = int((np.array(g_emin).mean() + g_min) / 2)
                                if len(b_emax):
                                    b_max = int((np.array(b_emax).mean() + b_max) / 2)

                                #pos_cand = int((r_min + g_min + b_max) / 3)
                                #pos_cand = int((r_min+g_min)/2)
                                pos_cand = r_min
                                pilot_list.append(pos_cand)
                                pilot_barcode[2] = and_image
                                rx_pilot[2] = GetColor(and_image, y_left_top, y_right_bottom, pos_cand)
                                if isPrintGraph == True:
                                    GetGraph(and_image, y_left_top, y_right_bottom, pos_cand, idx)
                                print("Synchronized!!")
                                flag_sync=True

                                ''' RGB Pilot 포지션을 모두 동일한 위치로 지정해서 사용 '''
                                pos = int(np.array(pilot_list).mean())
                                # rx_pilot[0] = GetColor(pilot_barcode[0], y_left_top, y_right_bottom, pos)
                                # rx_pilot[1] = GetColor(pilot_barcode[1], y_left_top, y_right_bottom, pos)
                                # rx_pilot[2] = GetColor(pilot_barcode[2], y_left_top, y_right_bottom, pos)
                                #
                                # GetGraph(pilot_barcode[0], y_left_top, y_right_bottom, pos)
                                # GetGraph(pilot_barcode[1], y_left_top, y_right_bottom, pos)
                                # GetGraph(pilot_barcode[2], y_left_top, y_right_bottom, pos)

                                print(np.array(rx_pilot).T)
                                channel_matrix=getChannelMatrix(np.array(rx_pilot).T)

                                print("Data Frame Position:{0}".format(pos))
                            elif (prev_frame==5 or prev_frame==6) and frame==7:
                                print("Receiving Data Frames")

                                r, g, b = GetColor(and_image, y_left_top, y_right_bottom, pos)
                                if isPrintGraph == True:
                                    GetGraph(and_image, y_left_top, y_right_bottom, pos, idx)
                                rx_color = estimating(r, g, b, channel_matrix)
                                rx_data = decoding(rx_color)
                                rx_data_list.append(rx_data)

                                frame_cnt += 1
                                idx += 1

                                if frame_cnt == data_size:
                                    pos = 0
                                    frame_cnt = 0
                                    flag_sync = False
                                    prev_frame = -1
                                    pilot_list = []
                                    rx_pilot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                                    pilot_barcode = [[], [], []]
                                    channel_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                                    pilot_list.clear()

                                    fe, be = calculateBER(origin_data_list, rx_data_list)
                                    rx_data_list.clear()
                                    print("FE: {0}, BE: {1}".format(fe, be))
                                    be_list.append(be)
                                    fe_list.append(fe)
                                    packet_cnt += 1
                            else:
                                print("Missing Frame: ", frame)
                                pos = 0
                                frame_cnt = 0
                                flag_sync = False
                                prev_frame = -1
                                pilot_list = []
                                rx_pilot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                                pilot_barcode = [[], [], []]
                                channel_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                                pilot_list.clear()
                else:
                    print("Non-SYNC")
                    pos = 0
                    frame_cnt = 0
                    flag_sync = False
                    prev_frame = -1
                    pilot_list = []
                    rx_pilot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                    pilot_barcode = [[], [], []]
                    channel_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                    pilot_list.clear()
                idx += 1
        else:
            break

        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyWindow(winName)
            break

    fe_list = np.array(fe_list)
    be_list = np.array(be_list)
    fe_sum = np.sum(fe_list)
    be_sum = np.sum(be_list)
    print(fe_list)
    print(be_list)
    print(fe_sum, be_sum)
    print(packet_cnt * 60)
