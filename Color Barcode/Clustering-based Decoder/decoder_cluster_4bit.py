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

# 4bit
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
origin_bit_list = np.array(list(bit_dict.keys()))
print(origin_bit_list)

# pilot의 Hue Color list
pilot_h_list = [Color.RED.value, Color.REDCOM.value, Color.GREEN.value, Color.GREENCOM.value, Color.BLUE.value,
                Color.BLUECOM.value]

# RR`GG`BB`(Pilot)에 대한 각각의 임계값(R/G/B) 지정
pilot_range = np.array([[150, 50, 55], [45, 135, 110], [45, 160, 50], [120, 45, 130], [40, 40, 170], [140, 135, 35]])

# Pilot의 Hue Color를 구하는 함수
def makePilotH():
    h_list = []
    for item in pilot_h_list:
        h, l, s = colorsys.rgb_to_hls(item[0], item[1], item[2])
        h_list.append(h)

    return h_list

# Cluster를 탐색하는 함수
# th : 임계값
# c : Color(1:R, 2:G, 3:B)
# start : start of cluster
# end : end of cluster
# ud : 1(임계값 이상인 부분을 추출), 2(임계값 이하의 부분을 추출)
def findCluster(th, c, start, end, ud):
    s = -1
    e = -1

    if ud == 1:  # upper
        if c == 1:  # Red
            for idx in range(start, end):
                if r_mean[idx] >= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if r_mean[idx] >= th:
                    e = idx
                    break

        elif c == 2: # Green
            for idx in range(start, end):
                if g_mean[idx] >= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if g_mean[idx] >= th:
                    e = idx
                    break
        else: # Blue
            for idx in range(start, end):
                if b_mean[idx] >= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if b_mean[idx] >= th:
                    e = idx
                    break

    else:  # Down
        if c == 1:  # Red
            for idx in range(start, end):
                if r_mean[idx] <= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if r_mean[idx] <= th:
                    e = idx
                    break

        elif c == 2: # Green
            for idx in range(start, end):
                if g_mean[idx] <= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if g_mean[idx] <= th:
                    e = idx
                    break

        else: # Blue
            for idx in range(start, end):
                if b_mean[idx] <= th:
                    s = idx
                    break

            for idx in range(end-1, start-1, -1):
                if b_mean[idx] <= th:
                    e = idx
                    break

    return s, e

# Clutser 구하는 함수
# Cluster는 각 RR`GG`BB` 모두 임계값에 대한 설정만 다르고 나머지 동일
# Red, Green, Blue 3색의 임계값을 충족하는 Cluster를 생성
def findRCluster(): # R Cluster
    s, e = findCluster(pilot_range[0][0],1,0,bar_length,1)
    # print(s,e)
    s, e = findCluster(pilot_range[0][1],2,s,e,2)
    # print(s,e)
    s, e = findCluster(pilot_range[0][2],3,s,e,2)
    return s, e

def findRCompCluster(): # R_Comp Cluster
    s, e = findCluster(pilot_range[1][0],1,0,bar_length,2)
    # print(s,e)
    s, e = findCluster(pilot_range[1][1],2,s,e,1)
    # print(s,e)
    s, e = findCluster(pilot_range[1][2],3,s,e,1)
    return s, e

def findGCluster(): # G Cluster
    s, e = findCluster(pilot_range[2][0],1,0,bar_length,2)
    # print(s,e)
    s, e = findCluster(pilot_range[2][1],2,s,e,1)
    # print(s,e)
    s, e = findCluster(pilot_range[2][2],3,s,e,2)
    return s, e

def findGCompCluster(): # G_Comp Cluster
    s, e = findCluster(pilot_range[3][0],1,0,bar_length,1)
    # print(s,e)
    s, e = findCluster(pilot_range[3][1],2,s,e,2)
    # print(s,e)
    s, e = findCluster(pilot_range[3][2],3,s,e,1)
    return s, e

def findBCluster(): # B Cluster
    s, e = findCluster(pilot_range[4][0],1,0,bar_length,2)
    # print(s,e)
    s, e = findCluster(pilot_range[4][1],2,s,e,2)
    # print(s,e)
    s, e = findCluster(pilot_range[4][2],3,s,e,1)
    return s, e

def findBCompCluster(): # B Comp_Cluster
    s, e = findCluster(pilot_range[5][0],1,0,bar_length,1)
    # print(s,e)
    s, e = findCluster(pilot_range[5][1],2,s,e,1)
    # print(s,e)
    s, e = findCluster(pilot_range[5][2],3,s,e,2)
    return s, e

# Frame의 특정 위치의 Hue Color를 계산하는 함수
def getHue(pos):
    h, l, s = colorsys.rgb_to_hls(r_mean[pos], g_mean[pos], b_mean[pos])
    return h


# channel matrix 구하는 부분
# rx_matrix : Pilot의 R, G, B를 행렬로 표현
def getChannelMatrix(rx_matrix):
    rx_matrix = rx_matrix.T

    # 본래 전치행렬을 지정해야 맞지만, RGB는 배치가 동일하여 그냥 사용
    origin_matrix = [Color.RED.value, Color.GREEN.value, Color.BLUE.value]
    channel_matrix = np.dot(rx_matrix, np.linalg.inv(origin_matrix))
    print(rx_matrix)
    print(channel_matrix)

    for line in channel_matrix:
        np.savetxt(f, line, fmt='%.9f')

    return channel_matrix

# Channel Matrix를 사용해서 수신된 data의 R/G/B 값을 보정
# [Parameter]
# pos : RX frame에서의 위치
#
# [Return]
# color : 보정된 RGB 색상 [R, G, B]
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


# 보정된 R/G/B 값을 바탕으로 Decoding 실시
# [Parameter]
# color : [Red, Green, Blue], 보정된 색상값
#
# [Return]
# minIdx : decoding 결과(심볼값) 3bit(0~7), 4bit(0~15)
def decoding(color):
    est_h, est_l, est_s = colorsys.rgb_to_hls(color[0], color[1], color[2]) # RGB -> HLS

    # 심볼들의 Hue Color Value를 계산
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
        # h1, h2는 Hue Color Constellation에서 clockwise, counter-clockwise 방향에 따른 Hue Color 차이값
        h1 = abs(item - est_h)
        h2 = abs(1 + item - est_h)
        diff = 0
        if h1 < h2:
            diff = h1
        else:
            diff = h2

        # 근사값으로 추정
        if diff < min:
            min = diff
            minIdx = idx
        idx += 1

    return minIdx

# 중앙값을 가져오는 함수
def getMid(a, b):
    return int((a + b) / 2)


# 심볼에러(FE), 비트에러(BE) 계산
# [Paraemter]
# origin_data_list : TX된 원래의 신호(심볼)
# rx_data_list : RX된 신호(심볼)
#
# [Return]
# fe : 심볼 에러
# be : 비트 에러
def calculateBER(origin_data_list, rx_data_list):
    fe = 0
    be = 0

    print(origin_data_list)
    print(rx_data_list)

    f.writelines('{0} \n'.format(origin_data_list))
    f.writelines('{0} \n'.format(rx_data_list))

    # FER, BER
    for origin_frame, rx_frame in zip(origin_data_list, rx_data_list):
        if origin_frame != rx_frame: # 심볼에러 체크
            fe += 1
            origin_bit = origin_bit_list[origin_frame]
            rx_bit = origin_bit_list[rx_frame]

            for origin, rx in zip(origin_bit, rx_bit):
                if origin != rx: # 비트에러 체크
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


######################################################################
# Path, File List
path = "E:\\frames_yaw_20\\" # cutFrame으로 잘라놓은 결과가 저장된 경로
# path = './frames_115_4bit/'
file_list = os.listdir(path)
f = open("yaw_20_anal.txt", 'w')

size = len(file_list)  # size of File List, frame의 개수

img1 = cv2.imread(path + file_list[0])  # to get default size
height, width, channel = img1.shape

pilot_h = makePilotH() # Pilot의 Hue color list

# R/G/B
# Decoding
# # # 4bit
# origin_data_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
# origin_data_list = origin_data_list + [ 11, 12, 13, 14, 15 ,0, 1]
# np.array(origin_data_list)

# 3bit
origin_data_list = [1, 2, 3, 4, 5, 6, 7] * 3
origin_data_list = origin_data_list + [0, 1, 2, 3, 4, 5]
np.array(origin_data_list)


frame_list = []

# 패킷 카운터
packet_cnt = 0

# test
flag = 0  # 0:find pilot, 1:get Data

# 추정된 Pilot에 대한 값들을 저장하는 리스트
# 추정횟수, Hue Color 근사한 정도, 프레임 위치
p_list = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]

# 파일럿 카운터
# 파일럿을 추정하는 동안 계속 카운트가 진행되며, 이 값이 2 이상인 경우 파일럿이 아니라고 간주하고 그 때까지 추정한 값을 모두 버림
# 파일럿 프레임이면서 연속적인 형태를 나타나는 경우에는 항상 0으로 초기화하기 때문에 0 아니면 1일 때에만 유효한 프레임으로 간주
p_cnt = 0

# Pilot의 R, G, B 위치 저장
pos_pilot = []

# Sync 추정 상태에 관한 Flag, 1~6:R~B`, 7:Sync 파악 완료
last_pilot = 0

# 수신된 Pilot RGB의 각각의 Red/Blue/Green 값을 저장한 Matrix
# 이 matrix로 바로 Channel Matrix를 추정하면 안되고
# Channel Matrix 추정할 때는 Transpose 적용해야 함
rx_pilot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

# 수신된 데이터 저장
rx_data_list = []

# 패킷 당 비트 에러 저장
be_list = []

data_cnt = 0 # 데이터 프레임 카운터
data_pos = 0 # 데이터 프레임 위치

idx = 0 # plot, image 저장 위한 변수

# 특정 색상이 약하게 나타나는 경우 보정 필요할 수도 있음
# 보정이 진행되면 임계값도 보정해야 함
pilot_range[:, 0] = pilot_range[:, 0] + 0
pilot_range[:, 1] = pilot_range[:, 1] + 0
pilot_range[:, 2] = pilot_range[:, 2] + 0

# 전체 수신 프레임을 탐색
for file in file_list[:]:
    img = cv2.imread(path + file)

    # crop
    # 수신 프레임에서 Barcode 영역만 잘라내는 부분
    # Barcode 추출은 수작업으로 해야 함
    img_crop = img[48:1012, 1241:1257, :]

    # Barcode의 width, height, channel
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

    # Color Barcode의 위치별 R, G, B 평균 값 계산
    r_mean = rotated_r.mean(axis=1)[:].T + 0
    g_mean = rotated_g.mean(axis=1)[:].T + 0
    b_mean = rotated_b.mean(axis=1)[:].T + 0

    # 최대 최소 고려
    r_max = np.argmax(r_mean)
    r_min = np.argmin(r_mean)
    g_max = np.argmax(g_mean)
    g_min = np.argmin(g_mean)
    b_max = np.argmax(b_mean)
    b_min = np.argmin(b_mean)

    # barcode의 길이
    bar_length = len(r_mean)

    # plt 출력
    plt.subplot(411), plt.imshow(rotated_plt)
    # R, G, B graph
    plt.subplot(423), plt.imshow(rotated_r.T)
    plt.subplot(424), plt.plot(r_mean, color='r'), plt.title(str(r_mean[r_max])+' '+str(r_max)+' '+str(r_mean[r_min])+' '+str(r_min))

    plt.subplot(425), plt.imshow(rotated_g.T)
    plt.subplot(426), plt.plot(g_mean, color='g'), plt.title(str(g_mean[g_max])+' '+str(g_max)+' '+str(g_mean[g_min])+' '+str(g_min))

    plt.subplot(427), plt.imshow(rotated_b.T)
    plt.subplot(428), plt.plot(b_mean, color='b'), plt.title(str(b_mean[b_max])+' '+str(b_max)+' '+str(b_mean[b_min])+' '+str(b_min))
    plt.tight_layout()
    # plt.show()
    if idx >= 100:
        plt.savefig('./figures_yaw_20/' + str(idx) + '.png', dpi=300) # ./figures_110_4bit/
    elif idx >= 10:
        plt.savefig('./figures_yaw_20/0' + str(idx) + '.png', dpi=300)
    else:
        plt.savefig('./figures_yaw_20/00' + str(idx) + '.png', dpi=300) # E:\\figures_95_4bit\\
    plt.close()
    idx += 1

    cur_pos = 0
    vxline = []
    pilot_list = []

    # flag == 0이고, last_pilot < 7인 경우
    # pilot frame을 찾지 못한 상태일 때
    if flag == 0 and last_pilot < 7:
        if p_cnt >= 2: # 수신 프레임 3개에 걸쳐 연속적으로 같은 것이 나타나거나, 아무것도 나타나지 않은 경우
            # 관련 변수 모두 초기화
            p_list = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
            pos_pilot = []
            last_pilot = 0
            rx_pilot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            pilot_list.clear()
            vxline.clear()
            last_pilot = 0
            p_cnt = 0

        # Clustering한 결과에서 Cluster의 중앙값을 pilot frame 위치로 추정
        # Pilot Frame의 연속 패턴(RR`GG`BB`)이 나타날 경우, last_pilot은 1부터 6까지 계속 증가함
        # Pilot Red
        start, end = findRCluster()
        mid = getMid(start, end)
        print("Red: ", start, end, mid)
        if mid != -1: # mid가 -1이면, Threshold를 충족하는 cluster가 없는 것
            last_pilot = 1
            # Constellation 상에서의 근사치를 구하기 위해 clockwise, counter-clockwise 고려
            h1 = abs(pilot_h[0] - getHue(mid))
            h2 = abs(1 - pilot_h[0] - getHue(mid))
            h = 0
            if h1 < h2:
                h = h1
            else:
                h = h2

            # R Frame으로 추정되는 값에 대한 정보 갱신
            p_list[0][2] += 1 # R Frame 탐색 횟수
            p_list[0][1] = h # Hue 값
            p_list[0][0] = mid # R Frame Position
            pos_pilot.append(mid) # Pilot의 R, G, B position을 저장
            pilot_list.append(1)
            rx_pilot[0] = r_mean[mid], g_mean[mid], b_mean[mid] # Pilot의 R/G/B Frame은 Channel Matrix 추정 위해 따로 저장
            vxline.append([mid, 1])
            p_cnt = 0 # 연속 여부 처리하는 플래그 초기화

        # 이후 다른 Frame도 동일하게 작동
        # Pilot Red Comp
        start, end = findRCompCluster()
        mid = getMid(start, end)
        print("Red_Comp: ", start, end, mid)
        if mid != -1 and last_pilot == 1:
            h = abs(pilot_h[1] - getHue(mid))
            last_pilot = 2
            if p_list[1][1] > h:
                p_list[1][2] += 1
                p_list[1][1] = h
                p_list[1][0] = mid
                pos_pilot.append(mid)
                pilot_list.append(2)
                vxline.append([mid, 2])
                p_cnt = 0

        # Pilot Green
        start, end = findGCluster()
        mid = getMid(start, end)
        print("Green: ", start, end, mid)
        if mid != -1 and last_pilot == 2:
            h = abs(pilot_h[2] - getHue(mid))
            last_pilot = 3
            if p_list[2][1] > h:
                p_list[2][2] += 1
                p_list[2][1] = h
                p_list[2][0] = mid
                pos_pilot.append(mid)
                pilot_list.append(2)
                rx_pilot[1] = r_mean[mid], g_mean[mid], b_mean[mid]
                vxline.append([mid, 2])
                p_cnt = 0

        # Pilot Green Comp
        start, end = findGCompCluster()
        mid = getMid(start, end)
        print("Green_Comp: ", start, end, mid)
        if mid != -1 and last_pilot  == 3:
            h = abs(pilot_h[3] - getHue(mid))
            last_pilot = 4
            if p_list[3][1] > h:
                p_list[3][2] += 1
                p_list[3][1] = h
                p_list[3][0] = mid
                pos_pilot.append(mid)
                pilot_list.append(3)
                vxline.append([mid, 3])
                p_cnt = 0

        # Pilot Blue
        start, end = findBCluster()
        mid = getMid(start, end)
        print("Blue: ", start, end, mid)
        if mid != -1 and last_pilot == 4:
            h = abs(pilot_h[4] - getHue(mid))
            last_pilot = 5
            if p_list[4][1] > h:
                p_list[4][2] += 1
                p_list[4][1] = h
                p_list[4][0] = mid
                pos_pilot.append(mid)
                pilot_list.append(4)
                rx_pilot[2] = r_mean[mid], g_mean[mid], b_mean[mid]
                vxline.append([mid, 4])
                p_cnt = 0

        # Pilot Blue Comp
        start, end = findBCompCluster()
        mid = getMid(start, end)
        print("Blue_Comp: ", start, end, mid)
        if mid != -1 and last_pilot == 5:
            h = abs(pilot_h[5] - getHue(mid))
            last_pilot = 6
            if p_list[5][1] > h:
                p_list[5][2] += 1
                p_list[5][1] = h
                p_list[5][0] = mid
                pos_pilot.append(mid)
                pilot_list.append(5)
                vxline.append([mid, 5])
                p_cnt = 0

        print(last_pilot)
        print(p_list)

        if last_pilot == 6:
            last_pilot = 7

        if len(pilot_list) == 0:
            p_list = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
            pos_pilot = []
            last_pilot = 0
            rx_pilot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            pilot_list.clear()
            vxline.clear()
            last_pilot = 0
            p_cnt = 0
            print("This1")
            continue

        # Sync를 맞춘 경우
        # Channel Matrix(H) 추정
        # 만약 RX Frame에 B` Frame과 Data Frame이 같이 존재하는 경우에 대한 처리 진행
        if last_pilot == 7:
            channel_matrix = getChannelMatrix(np.array(rx_pilot)) # Channel Matrix

            half_length = int(bar_length) / 2

            # 파일럿의 포지션을 추정하는 부분
            p1, p2, p3 = p_list[0][0], p_list[2][0], p_list[4][0]

            # 수신 프레임의 끝과 앞에 걸치는 경우에 대한 처리
            # 단순히 평균을 구해버리면 아래와 같은 상황에 대해 문제 발생

            # 바코드 길이 : 1000, 파일럿 3개의 위치(900, 10, 900)
            # 단순 평균 : 1810/3 = 603.3, 3개의 위치와 아예 다른 지점을 추정하게 됨

            # 실제 파일럿 위치 10은 다음 수신 프레임 시작점에 걸친 것이기 때문에
            # 실질적으로 바코드의 길이를 더한 1010을 위치로 잡아줘야 이러한 문제 해결 가능
            # 걸치는 부분 고려한 평균 : 2810 / 3 = 936.7

            # 60fps 영상을 30fps로 수신했을 때, 수신 프레임에서의 프레임 개수를 고려,
            # 전체 바코드 길이의 반보다 더 큰 차이가 나면 다음 프레임에 걸쳤던 것으로 간주
            # 만약에 수신 영상 fps가 달라지면 이 부분도 수정이 필요
            # 판단 기준 : (Tx fps/Rx fps) * barcode_length

            sub1 = abs(p1 - p2)
            if sub1 > half_length:
                if p1 < p2:
                    p1 += bar_length
                else:
                    p2 += bar_length

            sub2 = abs(p2 - p3)
            if sub2 > half_length:
                if p2 < p3:
                    p2 += bar_length
                else:
                    p3 += bar_length

            sub3 = abs(p3 - p1)
            if sub3 > half_length:
                if p3 < p1:
                    p3 += bar_length
                else:
                    p1 += bar_length

            print(p1, p2, p3)

            # 보정 과정을 거친 파일럿 프레임의 위치를 사용,
            # data frame의 위치를 Pilot의 R/G/B 위치의 평균으로 지정
            data_pos = int(np.array([p1, p2, p3]).mean()) % bar_length

            print(data_pos)
            print("Last Pilot Pos: ", pos_pilot[len(pos_pilot) - 1])

            # pilot frame 마지막인 B`과 data frame이 수신 frame에 같이 존재하는 경우,
            # 해당 부분도 data이므로 decoding 진행
            if pos_pilot[len(pos_pilot) - 1] < data_pos:
                data_cnt += 1

                vxline.append([data_pos, 7])
                print("Data Pos: ", data_pos)
                rx_color = estimating(data_pos)
                rx_data = decoding(rx_color)
                rx_data_list.append(rx_data)

    # Data Frame이 수신되고 있는 경우,
    else:
        data_cnt += 1

        vxline.append([data_pos, 7])
        print(data_pos)
        rx_color = estimating(data_pos)
        rx_data = decoding(rx_color)
        rx_data_list.append(rx_data)

        # data 27개를 다 받아들였을 때,
        # 다음 패킷을 위해 관련 변수들 초기화
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
            pilot_list.clear()

            # 심볼 에러(FE)와 비트 에러(BE)를 계산
            fe, be = calculateBER(origin_data_list, rx_data_list)
            rx_data_list.clear()
            print("FE: {0}, BE: {1}".format(fe, be))
            be_list.append(be)

            # 수신 프레임 하나에서 Data Frame 마지막 + Pilot의 RR`이 나타나는 경우 처리
            start, end = findRCluster()
            mid = getMid(start, end)
            print("Red: ", start, end, mid)
            if mid != -1: # R 탐색
                last_pilot = 1
                h1 = abs(pilot_h[0] - getHue(mid))
                h2 = abs(1 - pilot_h[0] - getHue(mid))
                h = 0
                if h1 < h2:
                    h = h1
                else:
                    h = h2
                if p_list[0][1] > h:
                    p_list[0][2] += 1
                    p_list[0][1] = h
                    p_list[0][0] = mid
                    pos_pilot.append(mid)
                    pilot_list.append(1)
                    rx_pilot[0] = r_mean[mid], g_mean[mid], b_mean[mid]
                    vxline.append([mid, 1])
                    p_cnt = 0

            start, end = findRCompCluster()
            mid = getMid(start, end)
            print("Red_Comp: ", start, end, mid)
            if mid != -1 and last_pilot == 1: # R` 탐색
                h = abs(pilot_h[1] - getHue(mid))
                last_pilot = 2
                if p_list[1][1] > h:
                    p_list[1][2] += 1
                    p_list[1][1] = h
                    p_list[1][0] = mid
                    pos_pilot.append(mid)
                    pilot_list.append(2)
                    vxline.append([mid, 2])
                    p_cnt = 0

    p_cnt += 1

    # # plt 출력
    # # 선을 긋기 위해 뒀던 코드
    # plt.subplot(411), plt.imshow(rotated_plt)
    # # R, G, B graph
    # plt.subplot(423), plt.imshow(rotated_r.T)
    # plt.subplot(424), plt.plot(r_mean, color='r'), plt.title(
    #     str(r_mean[r_max]) + ' ' + str(r_max) + ' ' + str(r_mean[r_min]) + ' ' + str(r_min))
    # for x_, p in vxline:
    #     # if p % 2 == 1 and flag == 0:
    #     #     plt.axvline(x=x_, color='k', linewidth = 1)
    #     if p == 5:
    #         plt.axvline(x=x_, color='k', linestyle=':', linewidth=3)
    #     elif p < 5:
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
    #     if p == 5:
    #         plt.axvline(x=x_, color='k', linestyle=':', linewidth=3)
    #     elif p < 5:
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
    #     if p == 5:
    #         plt.axvline(x=x_, color='k', linestyle=':', linewidth=3)
    #     elif p < 5:
    #         plt.axvline(x=x_, color='k', linewidth=1)
    #     else:
    #         plt.axvline(x=x_, color='r', linewidth=1)
    #
    # plt.tight_layout()
    # if idx >= 100:
    #     plt.savefig('./figures_90_3bit/' + str(idx) + '.png', dpi=300)
    # elif idx >= 10:
    #     plt.savefig('./figures_90_3bit/0' + str(idx) + '.png', dpi=300)
    # else:
    #     plt.savefig('./figures_90_3bit/00' + str(idx) + '.png', dpi=300)
    # plt.close()

    # idx += 1

# 수신 성공한 패킷들의 비트 에러의 합을 계산
be_list = np.array(be_list)
be_sum = np.sum(be_list)
print(be_sum) # 총 비트 에러
print(packet_cnt * 60) # 프레임의 수

f.writelines('SUM_BitError: {0} \n'.format(be_sum))
f.writelines('SUM_FRAMES(1packet = 60f): {0} \n'.format(packet_cnt * 60))