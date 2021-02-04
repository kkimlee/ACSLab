import cv2
import colorsys  # rgb to hsl
import os
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


# RGB
# openCV는 BGR임을 유의
class Color(Enum):
    RED = [255, 0, 0]
    REDCOM = [0, 255, 255]
    GREEN = [0, 255, 0]
    GREENCOM = [255, 0, 255]
    BLUE = [0, 0, 255]
    BLUECOM = [255, 255, 0]


class ColorSet(Enum):
    C0000 = [255,0,0]
    C0001 = [255,95,0]
    C0011 = [255,191,0]
    C0010 = [223,255,0]
    C0110 = [127,255,0]
    C0111 = [31,255,0]
    C0101 = [0,255,63]
    C0100 = [0,255,159]
    C1100 = [0,255,255]
    C1101 = [0,159,255]
    C1111 = [0,63,255]
    C1110 = [31,0,255]
    C1010 = [127,0,255]
    C1011 = [223,0,255]
    C1001 = [255,0,191]
    C1000 = [255,0,95]

# bit_list
bit_dict = {
    "0000": ColorSet.C0000.value,
    "0001": ColorSet.C0001.value,
    "0011": ColorSet.C0011.value,
    "0010": ColorSet.C0010.value,
    "0110": ColorSet.C0110.value,
    "0111": ColorSet.C0111.value,
    "0101": ColorSet.C0101.value,
    "0100": ColorSet.C0100.value,
    "1100": ColorSet.C1100.value,
    "1101": ColorSet.C1101.value,
    "1111": ColorSet.C1111.value,
    "1110": ColorSet.C1110.value,
    "1010": ColorSet.C1010.value,
    "1011": ColorSet.C1011.value,
    "1001": ColorSet.C1001.value,
    "1000": ColorSet.C1000.value
}

origin_bit_list = np.array(list(bit_dict.keys()))
print(origin_bit_list)

# pilot_range를 지정해서
# pilot frame을 추적
pilot_range = np.array([[170, 50, 70], [40, 120, 130], [20, 190, 40], [130, 60, 145], [40, 40, 195], [150, 160, 50]])

pilot_h_list = [Color.RED.value, Color.REDCOM.value, Color.GREEN.value, Color.GREENCOM.value, Color.BLUE.value,
                Color.BLUECOM.value]


def makePilotH():
    h_list = []
    for item in pilot_h_list:
        h, l, s = colorsys.rgb_to_hls(item[0], item[1], item[2])
        h_list.append(h)

    return h_list


def findCluster(th, c, start, end, ud):
    s = -1
    e = -1

    if ud == 1:  # upper
        if c == 1:  # R
            for idx in range(start, end):
                if r_mean[idx] >= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if r_mean[idx] >= th:
                    e = idx
                    break

        elif c == 2:
            for idx in range(start, end):
                if g_mean[idx] >= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if g_mean[idx] >= th:
                    e = idx
                    break
        else:
            for idx in range(start, end):
                if b_mean[idx] >= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if b_mean[idx] >= th:
                    e = idx
                    break

    else:  # Down
        if c == 1:  # R
            for idx in range(start, end):
                if r_mean[idx] <= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if r_mean[idx] <= th:
                    e = idx
                    break

        elif c == 2:
            for idx in range(start, end):
                if g_mean[idx] <= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if g_mean[idx] <= th:
                    e = idx
                    break

        else:
            for idx in range(start, end):
                if b_mean[idx] <= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if b_mean[idx] <= th:
                    e = idx
                    break

    return s, e

def findRCluster():
    s, e = findCluster(pilot_range[0][0],1,0,bar_length,1)
    # print(s,e)
    s, e = findCluster(pilot_range[0][1],2,s,e,2)
    # print(s,e)
    s, e = findCluster(pilot_range[0][2],3,s,e,2)
    return s, e

def findRCompCluster():
    s, e = findCluster(pilot_range[1][0],1,0,bar_length,2)
    # print(s,e)
    s, e = findCluster(pilot_range[1][1],2,s,e,1)
    # print(s,e)
    s, e = findCluster(pilot_range[1][2],3,s,e,1)
    return s, e

def findGCluster():
    s, e = findCluster(pilot_range[2][0],1,0,bar_length,2)
    # print(s,e)
    s, e = findCluster(pilot_range[2][1],2,s,e,1)
    # print(s,e)
    s, e = findCluster(pilot_range[2][2],3,s,e,2)
    return s, e

def findGCompCluster():
    s, e = findCluster(pilot_range[3][0],1,0,bar_length,1)
    # print(s,e)
    s, e = findCluster(pilot_range[3][1],2,s,e,2)
    # print(s,e)
    s, e = findCluster(pilot_range[3][2],3,s,e,1)
    return s, e

def findBCluster():
    s, e = findCluster(pilot_range[4][0],1,0,bar_length,2)
    # print(s,e)
    s, e = findCluster(pilot_range[4][1],2,s,e,2)
    # print(s,e)
    s, e = findCluster(pilot_range[4][2],3,s,e,1)
    return s, e

def findBCompCluster():
    s, e = findCluster(pilot_range[5][0],1,0,bar_length,1)
    # print(s,e)
    s, e = findCluster(pilot_range[5][1],2,s,e,1)
    # print(s,e)
    s, e = findCluster(pilot_range[5][2],3,s,e,2)
    return s, e

def getHue(pos):
    h, l, s = colorsys.rgb_to_hls(r_mean[pos], g_mean[pos], b_mean[pos])
    return h


# channel matrix 구하는 부분
# RGB Pilot Frame
def getChannelMatrix(rx_matrix):
    origin_matrix = [Color.RED.value, Color.GREEN.value, Color.BLUE.value]
    channel_matrix = np.dot(rx_matrix, np.linalg.inv(origin_matrix))
    print(rx_matrix)
    print(channel_matrix)

    for line in channel_matrix:
        np.savetxt(f, line, fmt='%.9f')

    return channel_matrix


def estimating(pos):
    color = np.array([r_mean[pos], g_mean[pos], b_mean[pos]])

    print("Rx")
    print(color)

    # dot H
    # print(channel_matrix)
    color = np.dot(np.linalg.inv(channel_matrix), color)
    # color = np.clip(color, 0 , 255)

    print("Est")
    print(color)

    return color


def decoding(color):
    est_h, est_l, est_s = colorsys.rgb_to_hls(color[0], color[1], color[2])
    # print(color)
    # print(est_h,est_s,est_l)
    print(est_h)

    code_hls_list = []
    for item in ColorSet:
        h, l, s = colorsys.rgb_to_hls(item.value[0], item.value[1], item.value[2])
        code_hls_list.append(h)

    code_hls_list = np.array(code_hls_list)

    # decoding, maximum likelihood
    min = 1.0
    minIdx = -1
    idx = 0
    for item in code_hls_list:
        diff = abs(item - est_h)
        if diff < min:
            min = diff
            minIdx = idx
        idx += 1

    return minIdx


def getMid(a, b):
    return int((a + b) / 2)


# Calcualte FER, BER
def calculateBER(origin_data_list, rx_data_list):
    fe = 0
    be = 0

    print(origin_data_list)
    print(rx_data_list)

    f.writelines('{0} \n'.format(origin_data_list))
    f.writelines('{0} \n'.format(rx_data_list))

    # FER, BER
    for origin_frame, rx_frame in zip(origin_data_list, rx_data_list):
        if origin_frame != rx_frame:
            fe += 1
            origin_bit = origin_bit_list[origin_frame]
            rx_bit = origin_bit_list[rx_frame]

            for origin, rx in zip(origin_bit, rx_bit):
                if origin != rx:
                    be += 1

    f.writelines('FE: {0} \n'.format(fe))
    f.writelines('BE: {0} \n'.format(be))

    return fe, be


def getFrameGap(li):
    gap = []
    if li[0] > li[1]:
        gap.append(bar_length - li[0] + li[1])
    else:
        gap.append(li[1] - li[0])

    if li[2] > li[3]:
        gap.append(bar_length - li[2] + li[3])
    else:
        gap.append(li[3] - li[2])

    if li[4] > li[5]:
        gap.append(bar_length - li[4] + li[5])
    else:
        gap.append(li[5] - li[4])

    return int(np.array(gap).mean())


#################################
# Path, File List
path = "E:\\frames_95_4bit\\"
# path = './frames_110_3bit/'
file_list = os.listdir(path)
f = open("95_4bit_anal.txt", 'w')

size = len(file_list)  # size of list

img1 = cv2.imread(path + file_list[0])  # to get default size
height, width, channel = img1.shape

pilot_h = makePilotH()

# R/G/B
# Decoding
# 3bit data frame
# 27 data + 27 comp

# origin_data_list = [ 2, 4, 3, 6, 3, 5 ] * 4
# origin_data_list = origin_data_list + [ 1, 7, 3 ]
# np.array(origin_data_list)

# origin_data_list = [ 7, 4, 7, 4, 7, 4 ] * 4
# origin_data_list = origin_data_list + [ 7, 4, 7 ]
# np.array(origin_data_list)

# 4bit
origin_data_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
origin_data_list = origin_data_list + [ 11, 12, 13, 14, 15 ,0, 1]
np.array(origin_data_list)

# # 3bit
# origin_data_list = [1, 2, 3, 4, 5, 6, 7] * 3
# origin_data_list = origin_data_list + [0, 1, 2, 3, 4, 5]
# np.array(origin_data_list)

frame_list = []

# packet_cnt
packet_cnt = 0


for data_pos in range(400,520,10):

    # test
    flag = 0  # 0:find pilot, 1:get Data
    flag_pilot = 0  # 0: Nothing, 1:R, 3:G, B:5, 7:Done
    cnt = 0  # 6:pilot frames
    cnt_list = []

    p_list = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
    pos_pilot = []
    last_pilot = 0
    rx_pilot = []
    frame_cnt = 0

    rx_data_list = []
    be_list = []

    idx = 0
    data_cnt = 0
    p_cnt = 0
    p_prev = 0

    pilot_range[:, 0] = pilot_range[:, 0] + 60
    pilot_range[:, 1] = pilot_range[:, 1] + 0
    pilot_range[:, 2] = pilot_range[:, 2] + 0

    pos = [500,460,500]
    pIdx = 0
    for file in file_list[122:125]:
        img = cv2.imread(path + file)
        # crop
        img_crop = img[83:1008, 1205:1216, :]

        width, height, channel = img_crop.shape

        print(file)
        # Rotate
        # 수직으로 찍힌 바코드를 수평으로 되돌리기, 전치행렬 사용
        rotated = img_crop

        # B, G, R 성분 분화
        rotated_b, rotated_g, rotated_r = cv2.split(rotated)

        # R, G, B 평균 값 계산
        r_mean = rotated_r.mean(axis=1)[:].T + 60
        g_mean = rotated_g.mean(axis=1)[:].T + 0
        b_mean = rotated_b.mean(axis=1)[:].T + 0

        rx_pilot.append([r_mean[pos[pIdx]], g_mean[pos[pIdx]], b_mean[pos[pIdx]]])
        last_pilot = 7
        pIdx += 1

    channel_matrix = getChannelMatrix(np.array(rx_pilot))
    # data_pos = [480, 840, 100, 450, 450, 480, 120, 500, 600, 100, 600, 600, 400, 600, 200, 600, 600, 700, 700, 700, 420,
    #             550, 550, 600, 600, 600, 600]

    data_pos = 480

    for file in file_list[125:152]:
        img = cv2.imread(path + file)
        # crop
        img_crop = img[83:1008, 1205:1216, :]

        width, height, channel = img_crop.shape

        print(file)
        # Rotate
        # 수직으로 찍힌 바코드를 수평으로 되돌리기, 전치행렬 사용
        rotated = img_crop

        # B, G, R 성분 분화
        rotated_b, rotated_g, rotated_r = cv2.split(rotated)

        # RGB graph 그리기
        # pyplot을 위해 RGB로 변경
        rotated_plt = cv2.merge([rotated_r.T, rotated_g.T, rotated_b.T])

        # 변형 결과 출력
        cv2.imshow('rotated', rotated)
        # plt.imshow(rotated_plt)
        # plt.show()

        # R, G, B 평균 값 계산
        r_mean = rotated_r.mean(axis=1)[:].T + 60
        g_mean = rotated_g.mean(axis=1)[:].T + 0
        b_mean = rotated_b.mean(axis=1)[:].T + 0

        data_cnt += 1

        print(data_pos)
        rx_color = estimating(data_pos)
        rx_data = decoding(rx_color)
        rx_data_list.append(rx_data)

        if data_cnt >= 27:
            packet_cnt += 1

            flag = 0
            data_cnt = 0
            frame_gap = 0
            data_pos = 0
            last_pilot = 0
            p_list = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
            pos_pilot = []
            last_pilot = 0
            rx_pilot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

            fe, be = calculateBER(origin_data_list, rx_data_list)
            rx_data_list.clear()
            print("FE: {0}, BE: {1}".format(fe, be))
            be_list.append(be)

        p_cnt += 1

be_list = np.array(be_list)
be_sum = np.sum(be_list)
print(be_sum)
print(packet_cnt * 60)

f.writelines('SUM_BitError: {0} \n'.format(be_sum))
f.writelines('SUM_FRAMES(1packet = 60f): {0} \n'.format(packet_cnt * 60))