import cv2
import colorsys # rgb to hsl
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

# 3bit
class ColorSet(Enum):
    C000 = [255,0,0]
    C001 = [255,191,0]
    C011 = [127,255,0]
    C010 = [0,255,63]
    C110 = [0,255,255]
    C111 = [0,63,255]
    C101 = [127,0,255]
    C100 = [255,0,191]

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
print(origin_bit_list)

# pilot_range를 지정해서
# pilot frame을 추적
pilot_range = np.array([[200, 80, 100], [95,150,150], [60, 210, 100], [110, 135, 180], [75, 80, 210], [180, 200, 65]])
# pilot_range = np.array([[200, 80, 100], [60,200,175], [80, 210, 100], [140, 90, 180], [40, 70, 220], [180, 200, 70]])

pilot_h_list = [Color.RED.value, Color.REDCOM.value, Color.GREEN.value, Color.GREENCOM.value, Color.BLUE.value, Color.BLUECOM.value]

def makePilotH():
    h_list = []
    for item in pilot_h_list:
        h, l, s = colorsys.rgb_to_hls(item[0], item[1], item[2])
        h_list.append(h)

    return h_list

def findR(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # color = estimating([r,g,b])
    if r >= pilot_range[cnt,0] and g <= pilot_range[cnt,1] and b <= pilot_range[cnt,2]:
        return 1
    return 0
def findRCluster():
    start_idx = -1
    end_idx = -1
    for idx in range(0, bar_length):
        if findR(0, idx) == 1:
            if start_idx < 0:
                start_idx = idx
            else:
                end_idx = idx
        else:
            if start_idx >= 0:
                end_idx = idx
                break
    return start_idx, end_idx

def findRComp(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # r, g, b = estimating([r,g,b])
    if r <= pilot_range[cnt,0] and g >= pilot_range[cnt,1] and b >= pilot_range[cnt,2]:
        return 1
    return 0
def findRCompCluster():
    start_idx = -1
    end_idx = -1
    for idx in range(0, bar_length):
        if findRComp(1, idx) == 1:
            if start_idx < 0:
                start_idx = idx
            else:
                end_idx = idx
        else:
            if start_idx >= 0:
                end_idx = idx
                break
    return start_idx, end_idx

def findG(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # r, g, b = estimating([r,g,b])
    if r <= pilot_range[cnt,0] and g >= pilot_range[cnt,1] and b <= pilot_range[cnt,2]:
        return 1
    return 0
def findGCluster():
    start_idx = -1
    end_idx = -1
    for idx in range(0, bar_length):
        if findG(2, idx) == 1:
            if start_idx < 0:
                start_idx = idx
            else:
                end_idx = idx
        else:
            if start_idx >= 0:
                end_idx = idx
                break
    return start_idx, end_idx

def findGComp(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # r, g, b = estimating([r,g,b])
    if r >= pilot_range[cnt,0] and g <= pilot_range[cnt,1] and b >= pilot_range[cnt,2]:
        return 1
    return 0
def findGCompCluster():
    start_idx = -1
    end_idx = -1
    for idx in range(0, bar_length):
        if findGComp(3, idx) == 1:
            if start_idx < 0:
                start_idx = idx
            else:
                end_idx = idx
        else:
            if start_idx >= 0:
                end_idx = idx
                break
    return start_idx, end_idx

def findB(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # r, g, b = estimating([r,g,b])
    if r <= pilot_range[cnt,0] and g <= pilot_range[cnt,1] and b >= pilot_range[cnt,2]:
        return 1
    return 0
def findBCluster():
    start_idx = -1
    end_idx = -1
    for idx in range(0, bar_length):
        if findB(4, idx) == 1:
            if start_idx < 0:
                start_idx = idx
            else:
                end_idx = idx
        else:
            if start_idx >= 0:
                end_idx = idx
                break
    return start_idx, end_idx

def findBComp(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # r, g, b = estimating([r,g,b])
    if r >= pilot_range[cnt,0] and g >= pilot_range[cnt,1] and b <= pilot_range[cnt,2]:
        return 1
    return 0
def findBCompCluster():
    start_idx = -1
    end_idx = -1
    for idx in range(0, bar_length):
        if findBComp(5, idx) == 1:
            if start_idx < 0:
                start_idx = idx
            else:
                end_idx = idx
        else:
            if start_idx >= 0:
                end_idx = idx
                break
    return start_idx, end_idx

def findPilot(cnt):
    if cnt == 0: # finding R
        s, e = findRCluster()
        print(s,e)
        mid = getMid(s, e)
    elif cnt == 1:
        s, e = findRCompCluster()
        mid = getMid(s, e)
    elif cnt == 2:
        s, e = findGCluster()
        mid = getMid(s, e)
    elif cnt == 3:
        s, e =findGCompCluster()
        mid = getMid(s, e)
    elif cnt == 4:
        s, e = findBCluster()
        mid = getMid(s, e)
    elif cnt == 5:
        s, e = findBCompCluster()
        mid = getMid(s, e)

    return mid

def getHue(pos):
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
    est_h, est_l, est_s = colorsys.rgb_to_hls(color[0],color[1],color[2])

    # print(color)
    # print(est_h,est_s,est_l)

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
    return int((a+b)/2)

# Calcualte FER, BER
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

def getFrameGap(li):
    gap = []
    if li[0] > li[1]:
        gap.append(bar_length - li[0] + li[1])
    else:
        gap.append(li[1]-li[0])

    if li[2] > li[3]:
        gap.append(bar_length - li[2] + li[3])
    else:
        gap.append(li[3]-li[2])

    if li[4] > li[5]:
        gap.append(bar_length - li[4] + li[5])
    else:
        gap.append(li[5]-li[4])

    return int(np.array(gap).mean())


#################################
# Path, File List
path = "./frames_rx/"
file_list = os.listdir(path)

size = len(file_list) # size of list

img1 = cv2.imread(path+file_list[0]) # to get default size
height, width, channel = img1.shape

pilot_h = makePilotH()

# R/G/B
# Decoding
# 3bit data frame
# 27 data + 27 comp
#
origin_data_list = [ 2, 4, 3, 6, 3, 5 ] * 4
origin_data_list = origin_data_list + [ 1, 7, 3 ]
np.array(origin_data_list)

# origin_data_list = [ 7, 4, 7, 4, 7, 4 ] * 4
# origin_data_list = origin_data_list + [ 7, 4, 7 ]
# np.array(origin_data_list)

# # 4bit
# origin_data_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
# origin_data_list = origin_data_list + [ 11, 12, 13, 14, 15 ,0, 1]
# np.array(origin_data_list)

# 3bit
# origin_data_list = [1, 2, 3, 4, 5, 6, 7] * 3
# origin_data_list = origin_data_list + [0, 1, 2, 3, 4, 5]
# np.array(origin_data_list)

# file_list = ["frs011.png", "frs041.png", "frs071.png", "frs101.png"]
# file_list = ["frs012.png", "frs042.png", "frs072.png", "frs102.png"]
# file_list = ["frs013.png", "frs043.png", "frs073.png", "frs103.png"]

frame_list = []

# packet_cnt
packet_cnt = 0

# test
flag = 0 # 0:find pilot, 1:get Data
flag_pilot = 0 # 0: Nothing, 1:R, 3:G, B:5, 7:Done
cnt = 0 # 6:pilot frames
cnt_list = []

p_list = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]
pos_pilot = []
last_pilot = 0
rx_pilot = [[0,0,0],[0,0,0],[0,0,0]]
frame_cnt = 0

rx_data_list = []
be_list = []

idx = 0
data_cnt = 0
data_pos = 0
p_cnt = 0
p_prev = 0

pilot_range[:, 0] = pilot_range[:, 0] + 20
pilot_range[:, 1] = pilot_range[:, 1]
pilot_range[:, 2] = pilot_range[:, 2]

for file in file_list[:]:
    img = cv2.imread(path+file)
    img_crop = img[40:1050, 1540:1550, :]

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
    r_mean = rotated_r.mean(axis=1)[:].T + 20
    g_mean = rotated_g.mean(axis=1)[:].T
    b_mean = rotated_b.mean(axis=1)[:].T

    # 최대 최소 고려
    r_max = np.argmax(r_mean)
    r_min = np.argmin(r_mean)
    g_max = np.argmax(g_mean)
    g_min = np.argmin(g_mean)
    b_max = np.argmax(b_mean)
    b_min = np.argmin(b_mean)

    bar_length = len(r_mean)

    # # plt 출력
    # plt.subplot(411), plt.imshow(rotated_plt)
    # # R, G, B graph
    # plt.subplot(423), plt.imshow(rotated_r.T)
    # plt.subplot(424), plt.plot(r_mean, color='r'), plt.title(str(r_mean[r_max])+' '+str(r_max)+' '+str(r_mean[r_min])+' '+str(r_min))
    #
    # plt.subplot(425), plt.imshow(rotated_g.T)
    # plt.subplot(426), plt.plot(g_mean, color='g'), plt.title(str(g_mean[g_max])+' '+str(g_max)+' '+str(g_mean[g_min])+' '+str(g_min))
    #
    # plt.subplot(427), plt.imshow(rotated_b.T)
    # plt.subplot(428), plt.plot(b_mean, color='b'), plt.title(str(b_mean[b_max])+' '+str(b_max)+' '+str(b_mean[b_min])+' '+str(b_min))
    # plt.tight_layout()
    # if idx >= 100:
    #     plt.savefig('./figures/' + str(idx) + '.png', dpi=300)
    # elif idx >= 10:
    #     plt.savefig('./figures/0' + str(idx) + '.png', dpi=300)
    # else:
    #     plt.savefig('./figures/00' + str(idx) + '.png', dpi=300)
    # plt.close()
    idx += 1

    cur_pos = 0
    # pilot frame을 찾지 못한 상태일 때,
    vxline = []
    pilot_list = []
    if flag == 0 and last_pilot < 7:
        start, end = findRCluster()
        mid = getMid(start, end)
        print("Red: ", start, end, mid)
        if mid != -1 and last_pilot <= 1:
            last_pilot = 1
            h1 = abs(pilot_h[0] - getHue(mid))
            h2 = abs(1 - pilot_h[0] - getHue(mid))
            h = 0
            if h1 < h2:
                h = h1
            else:
                h = h2
            if p_list[0][1] > h:
                p_list[0][1] = h
                p_list[0][0] = mid
                pos_pilot.append(mid)
                pilot_list.append(1)
                rx_pilot[0] = r_mean[mid], g_mean[mid], b_mean[mid]
                vxline.append([mid, 1])

        start, end = findRCompCluster()
        mid = getMid(start, end)
        print("Red_Comp: ", start, end, mid)
        if mid != -1 and last_pilot >= 1 and last_pilot <= 2:
            h = abs(pilot_h[1] - getHue(mid))
            last_pilot = 2
            if p_list[1][1] > h:
                p_list[1][1] = h
                p_list[1][0] = mid
                pos_pilot.append(mid)
                pilot_list.append(2)
                vxline.append([mid, 2])



        start, end = findGCluster()
        mid = getMid(start, end)
        print("Green: ", start, end, mid)
        if mid != -1 and last_pilot >= 2 and last_pilot <= 3:
            h = abs(pilot_h[2] - getHue(mid))
            last_pilot = 3
            if p_list[2][1] > h:
                p_list[2][1] = h
                p_list[2][0] = mid
                pos_pilot.append(mid)
                pilot_list.append(2)
                rx_pilot[1] = r_mean[mid], g_mean[mid], b_mean[mid]
                vxline.append([mid, 2])

        start, end = findGCompCluster()
        mid = getMid(start, end)
        print("Green_Comp: ", start, end, mid)
        if mid != -1 and last_pilot >= 3 and last_pilot <= 4:
            h = abs(pilot_h[3] - getHue(mid))
            last_pilot = 4
            if p_list[3][1] > h:
                p_list[3][1] = h
                p_list[3][0] = mid
                pos_pilot.append(mid)
                pilot_list.append(3)
                vxline.append([mid, 3])

        start, end = findBCluster()
        mid = getMid(start, end)
        print("Blue: ", start, end, mid)
        if mid != -1 and last_pilot >= 4 and last_pilot <= 5:
            h = abs(pilot_h[4] - getHue(mid))
            last_pilot = 5
            if p_list[4][1] > h:
                p_list[4][1] = h
                p_list[4][0] = mid
                pos_pilot.append(mid)
                pilot_list.append(4)
                rx_pilot[2] = r_mean[mid], g_mean[mid], b_mean[mid]
                vxline.append([mid, 4])

        start, end = findBCompCluster()
        mid = getMid(start, end)
        print("Blue_Comp: ", start, end, mid)
        if mid != -1 and last_pilot >= 5 and last_pilot <= 6:
            h = abs(pilot_h[5] - getHue(mid))
            last_pilot = 6
            if p_list[5][1] > h:
                p_list[5][1] = h
                p_list[5][0] = mid
                pos_pilot.append(mid)
                pilot_list.append(5)
                vxline.append([mid, 5])

        print(last_pilot)
        print(p_list)


        # B`까지 성공적으로 찾아낸 경우 관련 flag를 7로 변경
        if last_pilot == 6:
            last_pilot = 7

        # 추정된 값이 없으면 관련 변수 초기화
        if len(pilot_list) == 0:
            p_list = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
            pos_pilot = []
            last_pilot = 0
            rx_pilot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            pilot_list.clear()
            vxline.clear()
            last_pilot = 0
            p_cnt = 0
            print("This1")
            continue

        if last_pilot == 7:
            print(rx_pilot)
            print(np.array(rx_pilot).T)
            channel_matrix = getChannelMatrix(np.array(rx_pilot).T) #

            frame_gap = getFrameGap(pos_pilot)

            print("Frame Gap: ", frame_gap)
            print("Last Pilot Pos: ", pos_pilot[len(pos_pilot) - 1])
            if frame_gap + pos_pilot[len(pos_pilot) - 1] < bar_length:  # pilot frame 마지막과 data frame이 같이 있는 경우
                data_cnt += 1
                data_pos = pos_pilot[len(pos_pilot) - 1] + frame_gap

                vxline.append([data_pos, 7])
                print("Data Pos: ", data_pos)
                rx_color = estimating(data_pos)
                rx_data = decoding(rx_color)
                rx_data_list.append(rx_data)

            else:  # data frame이 다음 rx_frame에 나오는 경우,
                data_pos = (pos_pilot[len(pos_pilot) - 1] + frame_gap) % bar_length

    else:
        data_cnt += 1

        vxline.append([data_pos, 7])
        print(data_pos)
        rx_color = estimating(data_pos)
        rx_data = decoding(rx_color)
        rx_data_list.append(rx_data)

        if data_cnt >= 27:
            packet_cnt += 1

            flag = 0
            data_cnt = 0
            frame_gap = 0
            data_pos = -1
            last_pilot = 0
            p_list = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
            pos_pilot = []
            last_pilot = 0
            rx_pilot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            pilot_list.clear()

            fe, be = calculateBER(origin_data_list, rx_data_list)
            rx_data_list.clear()
            print("FE: {0}, BE: {1}".format(fe, be))
            be_list.append(be)

            # Data Frame 마지막 + Pilot의 R인 경우 처리
            start, end = findRCluster()
            mid = getMid(start, end)
            print("Red: ", start, end, mid)
            if mid != -1:
                pos_pilot.append(mid)
                pilot_list.append(1)
                rx_pilot.append([r_mean[mid], g_mean[mid], b_mean[mid]])
                vxline.append([mid, 1])

    p_cnt += 1

    # # plt 출력
    # plt.subplot(411), plt.imshow(rotated_plt)
    # # R, G, B graph
    # plt.subplot(423), plt.imshow(rotated_r.T)
    # plt.subplot(424), plt.plot(r_mean, color='r'), plt.title(
    #     str(r_mean[r_max]) + ' ' + str(r_max) + ' ' + str(r_mean[r_min]) + ' ' + str(r_min))
    # for x_, p in vxline:
    #     # if p % 2 == 1 and flag == 0:
    #     #     plt.axvline(x=x_, color='k', linewidth = 1)
    #     if p == 6:
    #         plt.axvline(x=x_, color='k', linestyle=':', linewidth=3)
    #     elif p < 6:
    #         plt.axvline(x=x_, color='k', linewidth=1)
    #     else:
    #         plt.axvline(x=x_, color='r', linewidth=1)
    #
    # plt.subplot(425), plt.imshow(rotated_g.T)
    # plt.subplot(426), plt.plot(g_mean, color='g'), plt.title(
    #     str(g_mean[g_max]) + ' ' + str(g_max) + ' ' + str(g_mean[g_min]) + ' ' + str(g_min))
    # for x_, p in vxline:
    #     # if p % 2 == 1 and flag == 0:
    #     #     plt.axvline(x=x_, color='k', linewidth = 1)
    #     if p == 6:
    #         plt.axvline(x=x_, color='k', linestyle=':', linewidth=3)
    #     elif p < 6:
    #         plt.axvline(x=x_, color='k', linewidth=1)
    #     else:
    #         plt.axvline(x=x_, color='r', linewidth=1)
    #
    # plt.subplot(427), plt.imshow(rotated_b.T)
    # plt.subplot(428), plt.plot(b_mean, color='b'), plt.title(
    #     str(b_mean[b_max]) + ' ' + str(b_max) + ' ' + str(b_mean[b_min]) + ' ' + str(b_min))
    # for x_, p in vxline:
    #     # if p % 2 == 1 and flag == 0:
    #     #     plt.axvline(x=x_, color='k', linewidth = 1)
    #     if p == 6:
    #         plt.axvline(x=x_, color='k', linestyle=':', linewidth=3)
    #     elif p < 6:
    #         plt.axvline(x=x_, color='k', linewidth=1)
    #     else:
    #         plt.axvline(x=x_, color='r', linewidth=1)
    #
    # plt.tight_layout()
    # plt.show()
    # # if idx >= 100:
    # #     plt.savefig('./figures/' + str(idx) + '.png', dpi=300)
    # # elif idx >= 10:
    # #     plt.savefig('./figures/0' + str(idx) + '.png', dpi=300)
    # # else:
    # #     plt.savefig('./figures/00' + str(idx) + '.png', dpi=300)
    # plt.close()

    # else:
    #     data_cnt += 1
    #     if data_cnt >= 26:
    #         flag = 0
    #         data_cnt = 0
    #         p_prev = 0
    #
    #     # plt 출력
    #     plt.subplot(411), plt.imshow(rotated_plt)
    #     # R, G, B graph
    #     plt.subplot(423), plt.imshow(rotated_r.T)
    #     plt.subplot(424), plt.plot(r_mean, color='r'), plt.title(str(r_mean[r_max])+' '+str(r_max)+' '+str(r_mean[r_min])+' '+str(r_min))
    #
    #     plt.subplot(425), plt.imshow(rotated_g.T)
    #     plt.subplot(426), plt.plot(g_mean, color='g'), plt.title(str(g_mean[g_max])+' '+str(g_max)+' '+str(g_mean[g_min])+' '+str(g_min))
    #
    #     plt.subplot(427), plt.imshow(rotated_b.T)
    #     plt.subplot(428), plt.plot(b_mean, color='b'), plt.title(str(b_mean[b_max])+' '+str(b_max)+' '+str(b_mean[b_min])+' '+str(b_min))
    #     plt.tight_layout()
    #     plt.savefig('./figures/'+str(idx)+'.png', dpi=300)
    #     plt.close()

be_list = np.array(be_list)
be_sum = np.sum(be_list)
print(be_sum)
print(packet_cnt * 60)