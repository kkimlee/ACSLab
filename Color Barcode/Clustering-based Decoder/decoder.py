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
pilot_range = np.array([[180, 70, 90], [70,170,170], [80, 180, 100], [130, 135, 170], [70, 70, 210], [170, 150, 50]])



def findR(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # color = estimating([r,g,b])`
    if r >= pilot_range[cnt,0] and g <= pilot_range[cnt,1] and b <= pilot_range[cnt,2]:
        return 1
    return 0

def findRComp(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # r, g, b  = estimating([r,g,b])
    if r <= pilot_range[cnt,0] and g >= pilot_range[cnt,1] and b >= pilot_range[cnt,2]:
        return 2
    return 0

def findG(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # r, g, b = estimating([r,g,b])
    if r <= pilot_range[cnt,0] and g >= pilot_range[cnt,1] and b <= pilot_range[cnt,2]:
        return 3
    return 0

def findGComp(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # r, g, b = estimating([r,g,b])
    print(r,g,b)
    if r >= pilot_range[cnt,0] and g <= pilot_range[cnt,1] and b >= pilot_range[cnt,2]:
        return 4
    return 0

def findB(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # r, g, b = estimating([r,g,b])
    if r <= pilot_range[cnt,0] and g <= pilot_range[cnt,1] and b >= pilot_range[cnt,2]:
        return 5
    return 0

def findBComp(cnt, idx):
    r, g, b = r_mean[idx], g_mean[idx], b_mean[idx]
    # r, g, b = estimating([r,g,b])
    if r >= pilot_range[cnt,0] and g >= pilot_range[cnt,1] and b <= pilot_range[cnt,2]:
        return 6
    return 0

def findPilot(cnt, idx):
    res = [findR(0, idx), findRComp(1, idx), findG(2, idx), findGComp(3, idx), findB(4, idx), findBComp(5, idx)]
    return res

# channel matrix 구하는 부분
# RGB Pilot Frame
def getH(file_list):
    cnt = 0
    rx_R = []
    rx_G = []
    rx_B = []

    for item in file_list:
        img = cv2.imread(item)

        img_crop = img[40:1050, 1350:1450, :]

        cv2.imshow("crop", img_crop)
        cv2.waitKey()

        # Rotate
        # 수직으로 찍힌 바코드를 수평으로 되돌리기
        rotated = img_crop

        # B, G, R 성분 분화
        rotated_b, rotated_g, rotated_r = cv2.split(rotated)

        # RGB graph 그리기
        # pyplot을 위해 RGB로 변경
        rotated_plt = cv2.merge([rotated_r.T, rotated_g.T, rotated_b.T])

        # 변형 결과 출력
        # cv2.imshow('rotated', rotated)
        # plt.imshow(rotated_plt)
        # plt.show()

        # R, G, B 평균 값 계산
        r_mean = rotated_r.mean(axis=1)[:].T + 30
        g_mean = rotated_g.mean(axis=1)[:].T
        b_mean = rotated_b.mean(axis=1)[:].T

        r_max = np.argmax(r_mean)
        r_min = np.argmin(r_mean)
        g_max = np.argmax(g_mean)
        g_min = np.argmin(g_mean)
        b_max = np.argmax(b_mean)
        b_min = np.argmin(b_mean)

        # plt.subplot(411), plt.imshow(rotated_plt)
        # # R, G, B graph
        # plt.subplot(423), plt.imshow(rotated_r.T)
        # plt.subplot(424), plt.plot(r_mean, color='r'), plt.title(str(r_mean[r_max]) + ' ' + str(r_max))
        # plt.subplot(425), plt.imshow(rotated_g.T)
        # plt.subplot(426), plt.plot(g_mean, color='g'), plt.title(str(g_mean[g_max]) + ' ' + str(g_max))
        # plt.subplot(427), plt.imshow(rotated_b.T)
        # plt.subplot(428), plt.plot(b_mean, color='b'), plt.title(str(b_mean[b_max]) + ' ' + str(b_max))
        # plt.tight_layout()
        # plt.show()

        # 0:R, 1:G, 2:B 순서로 나타남
        # 이는 영상 시작에 따라 다름
        if cnt%3 == 2:
            rx_B.append([r_mean[b_max],g_mean[b_max],b_mean[b_max]])
        elif cnt%3 == 0:
            # rx_R.append([r_mean[r_max],g_mean[r_max],b_mean[r_max]])
            rx_R.append([r_mean[b_min],g_mean[b_min],b_mean[b_min]])
        else:
            rx_G.append([r_mean[g_max],g_mean[g_max],b_mean[g_max]])

        cnt += 1

    rx_R = np.array(rx_R)
    rx_G = np.array(rx_G)
    rx_B = np.array(rx_B)

    rx_R_mean = rx_R.mean(axis=0)[:]
    rx_G_mean = rx_G.mean(axis=0)[:]
    rx_B_mean = rx_B.mean(axis=0)[:]

    origin_matrix = [Color.RED.value, Color.GREEN.value, Color.BLUE.value]
    origin_matrix = np.array(origin_matrix)

    rx_matrix = [rx_R_mean,rx_G_mean,rx_B_mean]
    rx_matrix = np.array(rx_matrix)

    print(rx_matrix)
    print(origin_matrix)

    channel_matrix = np.dot(rx_matrix, np.linalg.inv(origin_matrix))
    return channel_matrix

def getChannelMatrix(rx_matrix):
    origin_matrix = [Color.RED.value, Color.GREEN.value, Color.BLUE.value]
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

def findDataPosition(max_color, min_color):
    cand_pos = max_color
    cand_pos.sort()
    cand = cand_pos[1]
    margin = 50

    if np.abs(cand_pos[2]-cand) <= margin:
        cand_pos = min_color
        cand_pos.sort()
        cand = cand_pos[1]

    return cand

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
        # print(idx, minIdx, item, est_h)
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

#################################
# Path, File List
path = "./frames_90_3bit/"
file_list = os.listdir(path)

size = len(file_list) # size of list

img1 = cv2.imread(path+file_list[0]) # to get default size
height, width, channel = img1.shape

# get Channel Matrix
# R/G/B
pilot_list = [path+"frs011.png", path+"frs012.png", path+"frs013.png"]

# channel_matrix = getH(pilot_list)

frame_list = []

# Decoding
# 3bit data frame
# 27 data + 27 comp
#
# origin_data_list = [ 2, 4, 3, 6, 3, 5 ] * 4
# origin_data_list = origin_data_list + [ 1, 7, 3 ]

# origin_data_list = [ 7, 4, 7, 4, 7, 4 ] * 4
# origin_data_list = origin_data_list + [ 7, 4, 7 ]
# np.array(origin_data_list)

# # 4bit
# origin_data_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
# origin_data_list = origin_data_list + [ 11, 12, 13, 14, 15 ,0, 1]
# np.array(origin_data_list)

# 3bit
origin_data_list = [1, 2, 3, 4, 5, 6, 7] * 3
origin_data_list = origin_data_list + [0, 1, 2, 3, 4, 5]
np.array(origin_data_list)


min_pos = -1
min_be = 660

data_pos = 0
# test
flag = 0  # 0:find pilot, 1:get Data
cnt = 0  # 6:pilot frames
cnt_list = []

idx = 0

data_cnt = 0
frame_cnt = 0
rx_data_list = []
be_list = []

rx_R = []
rx_G = []
rx_B = []
rx_pilot = []

p_cnt = 0
p_prev = 0
p_prev_pos = 0
data_pos = 0
prev_frame_gap = 0
frame_gap = 0
pilot_pos = []

not_in_pilot = False

pilot_range[:, 0] = pilot_range[:, 0] + 20
pilot_range[:, 1] = pilot_range[:, 1] + 20

for file in file_list[:]:
    img = cv2.imread(path+file)
    img_crop = img[20:1032, 1080:1100, :]

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
    g_mean = rotated_g.mean(axis=1)[:].T + 20
    b_mean = rotated_b.mean(axis=1)[:].T


    # 최대 최소 고려
    r_max = np.argmax(r_mean)
    r_min = np.argmin(r_mean)
    g_max = np.argmax(g_mean)
    g_min = np.argmin(g_mean)
    b_max = np.argmax(b_mean)
    b_min = np.argmin(b_mean)

    bar_length = len(r_mean)

    # plt.subplot(411), plt.imshow(rotated_plt)
    # # R, G, B graph
    # plt.subplot(423), plt.imshow(rotated_r.T)
    # plt.subplot(424), plt.plot(r_mean, color='r'), plt.title(str(r_mean[r_max]) + ' ' + str(r_max))
    # plt.subplot(425), plt.imshow(rotated_g.T)
    # plt.subplot(426), plt.plot(g_mean, color='g'), plt.title(str(g_mean[g_max]) + ' ' + str(g_max))
    # plt.subplot(427), plt.imshow(rotated_b.T)
    # plt.subplot(428), plt.plot(b_mean, color='b'), plt.title(str(b_mean[b_max]) + ' ' + str(b_max))
    # plt.tight_layout()
    # plt.show()
    # if idx>=100:
    #     plt.savefig('./figures_747/'+str(idx)+'.png', dpi=300)
    # elif idx>=10:
    #     plt.savefig('./figures_747/0'+str(idx)+'.png',dpi=300)
    # else:
    #     plt.savefig('./figures_747/00'+str(idx)+'.png',dpi=300)
    # plt.close()

    pos_list = [r_min, r_max, b_min, b_max, g_min, g_max]
    pos_list.sort()

    cur_pos = 0
    # pilot frame을 찾지 못한 상태일 때,
    vxline = []
    if flag == 0:
        for pos in pos_list:
            r, g, b = r_mean[pos], g_mean[pos], b_mean[pos]

            nex = findPilot(cnt, pos)
            nex = np.array(nex)
            pos_pilot = np.sum(nex)
            print("Pilot:{0}".format(pos_pilot))

            if pos_pilot-p_prev == 1 and p_prev != 8:
                p_prev = pos_pilot
                vxline.append([pos, pos_pilot])
                if pos_pilot == 1:
                    rx_R = [r,g,b]
                    print(r,g,b)
                    p_prev_pos = pos
                    pilot_pos.append(pos)
                elif pos_pilot == 2:
                    frame_gap = np.abs(pos-p_prev_pos)
                    p_prev_pos = pos
                elif pos_pilot == 3:
                    rx_G = [r,g,b]
                    print(r,g,b)
                    p_prev_pos = pos
                    pilot_pos.append(pos)
                elif pos_pilot == 4:
                    prev_frame_gap = frame_gap
                    frame_gap = np.abs(pos-p_prev_pos)
                    p_prev_pos = pos
                elif pos_pilot == 5:
                    rx_B = [r,g,b]
                    prev_frame_gap = frame_gap
                    frame_gap = np.abs(pos-p_prev_pos)
                    p_prev_pos = pos
                    print(r,g,b)
                    pilot_pos.append(pos)
            else:
                pass

            if p_prev == 6: # pilot frame sync 완료
                flag = 1
                p_prev = 8

                data_pos = int(np.mean(np.array(pilot_pos)))

                print("frame gap", frame_gap, prev_frame_gap)

                # data_pos = findDataPosition([r_max, g_max, b_max], [r_min, g_min, b_min])
                rx_pilot = np.array([rx_R, rx_G, rx_B])

                channel_matrix = getChannelMatrix(rx_pilot)

                if data_pos < pos:
                    not_in_pilot = False
                    # data_pos = p_prev_pos
                    continue
                else:
                    not_in_pilot = True
                    data_cnt += 1

                # data_pos = findDataPosition([r_max, g_max, b_max], [r_min, g_min, b_min])
                vxline.append([data_pos, 7])
                print(data_pos)
                rx_color = estimating(data_pos)
                rx_data = decoding(rx_color)
                rx_data_list.append(rx_data)

            print("P_prev:{0}".format(p_prev))
            # print(cnt, nex)
            # cnt_list.append(cnt)
            # if np.abs(cnt-nex) > 1:
            #     cnt = 0
            # else:
            #     cnt = nex
            #     if cnt == 6: #find pilot
            #         flag = 1
            #         break

    else: # dataframe
        data_cnt += 1

        # data_pos = findDataPosition([r_max, g_max, b_max], [r_min, g_min, b_min])
        vxline.append([data_pos, 7])
        print(data_pos)
        rx_color = estimating(data_pos)
        rx_data = decoding(rx_color)
        rx_data_list.append(rx_data)

        if data_cnt >= 27:
            flag = 0
            data_cnt = 0
            frame_cnt += 1
            p_prev = 0
            data_pos = -1
            pilot_pos.clear()

            fe, be = calculateBER(origin_data_list, rx_data_list)
            rx_data_list.clear()
            print("FE: {0}, BE: {1}".format(fe,be))
            be_list.append(be)

    # plt 출력
    plt.subplot(411), plt.imshow(rotated_plt)
    # R, G, B graph
    plt.subplot(423), plt.imshow(rotated_r.T)
    plt.subplot(424), plt.plot(r_mean, color='r'), plt.title(str(r_mean[r_max])+' '+str(r_max)+' '+str(r_mean[r_min])+' '+str(r_min))
    for x_, p in vxline:
        # if p % 2 == 1 and flag == 0:
        #     plt.axvline(x=x_, color='k', linewidth = 1)
        if p == 6:
            plt.axvline(x=x_, color='k', linestyle=':', linewidth=3)
        elif p < 6:
            plt.axvline(x=x_ , color='k', linewidth=1)
        else:
            plt.axvline(x=x_, color='r', linewidth=1)

    plt.subplot(425), plt.imshow(rotated_g.T)
    plt.subplot(426), plt.plot(g_mean, color='g'), plt.title(str(g_mean[g_max])+' '+str(g_max)+' '+str(g_mean[g_min])+' '+str(g_min))
    for x_, p in vxline:
        # if p % 2 == 1 and flag == 0:
        #     plt.axvline(x=x_, color='k', linewidth = 1)
        if p == 6:
            plt.axvline(x=x_, color='k', linestyle=':', linewidth=3)
        elif p < 6:
            plt.axvline(x=x_ , color='k', linewidth=1)
        else:
            plt.axvline(x=x_, color='r', linewidth=1)

    plt.subplot(427), plt.imshow(rotated_b.T)
    plt.subplot(428), plt.plot(b_mean, color='b'), plt.title(str(b_mean[b_max])+' '+str(b_max)+' '+str(b_mean[b_min])+' '+str(b_min))
    for x_, p in vxline:
        # if p % 2 == 1 and flag == 0:
        #     plt.axvline(x=x_, color='k', linewidth = 1)
        if p == 6:
            plt.axvline(x=x_, color='k', linestyle=':', linewidth=3)
        elif p < 6:
            plt.axvline(x=x_ , color='k', linewidth=1)
        else:
            plt.axvline(x=x_, color='r', linewidth=1)

    plt.tight_layout()
    # plt.show()
    if idx>=100:
        plt.savefig('./figures_90_3bit/'+str(idx)+'.png', dpi=300)
    elif idx>=10:
        plt.savefig('./figures_90_3bit/0'+str(idx)+'.png',dpi=300)
    else:
        plt.savefig('./figures_90_3bit/00'+str(idx)+'.png',dpi=300)
    plt.close()

        # plt 출력
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
        # if idx>=100:
        #     plt.savefig('./figures/'+str(idx)+'.png', dpi=300)
        # elif idx>=10:
        #     plt.savefig('./figures/0'+str(idx)+'.png',dpi=300)
        # else:
        #     plt.savefig('./figures/00'+str(idx)+'.png',dpi=300)
        # plt.close()
    idx += 1

be_list = np.array(be_list)
be_sum = np.sum(be_list)
print(be_sum)
print(frame_cnt * 60)
# rx_red = np.array([r_mean[r_max],g_mean[r_max],b_mean[r_max]])
# est_red = np.dot(np.linalg.inv(channel_matrix), rx_red)
# # saturation
# est_red = np.clip(est_red, 0, 255)
#
# rx_green = np.array([r_mean[g_max],g_mean[g_max],b_mean[g_max]])
# est_green = np.dot(np.linalg.inv(channel_matrix), rx_green)
# est_green = np.clip(est_green, 0, 255)
#
# rx_blue = np.array([r_mean[b_max],g_mean[b_max],b_mean[b_max]])
# est_blue = np.dot(np.linalg.inv(channel_matrix), rx_blue)
# est_blue = np.clip(est_blue, 0, 255)
#
# print("RED")
# print(rx_red)
# print(est_red)
#
# print("GREEN")
# print(rx_green)
# print(est_green)
#
# print("BLUE")
# print(rx_blue)
# print(est_blue)

# check: pilot frame


# frame width = 995
# frs000, 540
# frs003, 476

# frs000(540) ~ frs003(x)
# convert rgb to hls
# h: 0.0 ~ 1.0 / 0~360
# l:
# s:

# BRG 순서로 나타남
# B 59.4 35.0 220.8
# R 197.6 59.8 64.4
# G 22.0 239.4 82.0


