import os
import cv2
import numpy as np
from enum import Enum
from scipy import signal
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import colorsys # rgb to hsl

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

def search(path, extension):
    files = os.listdir(path)
    
    file_list = list()
    for file in files:
        ext = os.path.splitext(file)[1]

        if ext == extension:
            file_list.append(file)
    
    return file_list

def compare(argmax, argmin, sort):
    # sort 이면 큰 수 출력
    if sort:
        if argmax > argmin:
            return argmax
        else:
            return argmin
    # sort 아니면 작은 수 출력
    else:
        if argmax < argmin:
            return argmax
        else:
            return argmin

def GetGraph(img, x_start, x_end, pos):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    red, green, blue = cv2.split(img)
    b_list = []
    g_list = []
    r_list = []
    for idx in range(x_start, x_end + 1):
        y_list = np.argwhere(blue[:, idx])
        if y_list.size > 0:
            y_start = y_list.min()
            y_end = y_list.max()
            b_list.append(blue[y_start:y_end + 1, idx].mean())
            g_list.append(green[y_start:y_end + 1, idx].mean())
            r_list.append(red[y_start:y_end + 1, idx].mean())

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
    

    
    plt.subplot(411), plt.imshow(img)
    
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
    
    plt.show()

def GetPixelGraph(img, x_start, x_end, frame_type):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    red, green, blue = cv2.split(img)
    b_list = []
    g_list = []
    r_list = []
    for idx in range(x_start, x_end):
        y_list = np.argwhere(blue[:, idx])
        if y_list.size > 0:
            y_start = y_list.min()
            y_end = y_list.max()
            b_list.append(blue[y_start:y_end + 1, idx].mean())
            g_list.append(green[y_start:y_end + 1, idx].mean())
            r_list.append(red[y_start:y_end + 1, idx].mean())


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


    # 최대 최소 고려
    r_max = np.argmax(r_mean_filtered)
    r_min = np.argmin(r_mean_filtered)
    g_max = np.argmax(g_mean_filtered)
    g_min = np.argmin(g_mean_filtered)
    b_max = np.argmax(b_mean_filtered)
    b_min = np.argmin(b_mean_filtered)

    
    # 최대 최소 위치 비교
    r_first = compare(r_max, r_min, 0)
    r_second = compare(r_max, r_min, 1)
    g_first = compare(g_max, g_min, 0)
    g_second = compare(g_max, g_min, 1)
    b_first = compare(b_max, b_min, 0)
    b_second = compare(b_max, b_min, 1)
    
    # 중간값 
    r_median = np.argsort(r_mean_filtered)[len(r_mean_filtered)//2]
    g_median = np.argsort(g_mean_filtered)[len(g_mean_filtered)//2]
    b_median = np.argsort(b_mean_filtered)[len(b_mean_filtered)//2]
    
    r_median = int((r_min + r_max)/2)
    g_median = int((g_min + g_max)/2)
    b_median = int((b_min + b_max)/2)
    
    # 파일럿 위치 추정
    pos = [r_median, g_median, b_median]
    
    # RR` 패턴
    R_1 = 0
    R_prime_1 = 0
    # XRR` 패턴
    R_2 = 0
    R_prime_2 = 0
    # R 찾기
    if r_mean_filtered[r_first] > r_mean_filtered[r_median] and \
       g_mean_filtered[g_first] < g_mean_filtered[g_median] and \
       b_mean_filtered[b_first] < b_mean_filtered[b_median]:
        R_1 = 1
    elif r_mean_filtered[r_second] > r_mean_filtered[r_median] and \
         g_mean_filtered[g_second] < g_mean_filtered[g_median] and \
         b_mean_filtered[b_second] < b_mean_filtered[b_median]:
          R_2 = 1
    if r_mean_filtered[r_second] < r_mean_filtered[r_median] and \
       g_mean_filtered[g_second] > g_mean_filtered[g_median] and \
       b_mean_filtered[b_second] > b_mean_filtered[b_median]:
        R_prime_1 = 1
    elif r_mean_filtered[r_first] < r_mean_filtered[r_median] and \
         g_mean_filtered[g_first] > g_mean_filtered[g_median] and \
         b_mean_filtered[b_first] > b_mean_filtered[b_median]:
          R_prime_2 = 1
    # if R and R_prime:
    #     print("RR`")
    
    # GG` 패턴
    G_1 = 0
    G_prime_1 = 0
    # R`GG` 패턴
    G_2 = 0
    G_prime_2 = 0
    # G 찾기
    if r_mean_filtered[r_first] < r_mean_filtered[r_median] and \
       g_mean_filtered[g_first] > g_mean_filtered[g_median] and \
       b_mean_filtered[b_first] < b_mean_filtered[b_median]:
        G_1 = 1
    elif r_mean_filtered[r_second] < r_mean_filtered[r_median] and \
         g_mean_filtered[g_second] > g_mean_filtered[g_median] and \
         b_mean_filtered[b_second] < b_mean_filtered[b_median]:
          G_2 = 1
    if r_mean_filtered[r_second] > r_mean_filtered[r_median] and \
       g_mean_filtered[g_second] < g_mean_filtered[g_median] and \
       b_mean_filtered[b_second] > b_mean_filtered[b_median]:
        G_prime_1 = 1
    elif r_mean_filtered[r_first] > r_mean_filtered[r_median] and \
         g_mean_filtered[g_first] < g_mean_filtered[g_median] and \
         b_mean_filtered[b_first] > b_mean_filtered[b_median]:
          G_prime_2 = 1
    # if G and G_prime:
    #     print("GG`")
    
    # BB` 패턴
    B_1 = 0
    B_prime_1 = 0
    # G`BB`X 패턴
    B_2 = 0
    B_prime_2 = 0
    # B 찾기
    if r_mean_filtered[r_first] < r_mean_filtered[r_median] and \
       g_mean_filtered[g_first] < g_mean_filtered[g_median] and \
       b_mean_filtered[b_first] > b_mean_filtered[b_median]:
        B_1 = 1
    elif r_mean_filtered[r_second] < r_mean_filtered[r_median] and \
         g_mean_filtered[g_second] < g_mean_filtered[g_median] and \
         b_mean_filtered[b_second] > b_mean_filtered[b_median]:
          B_2 = 1
    if r_mean_filtered[r_second] > r_mean_filtered[r_median] and \
       g_mean_filtered[g_second] > g_mean_filtered[g_median] and \
       b_mean_filtered[b_second] < b_mean_filtered[b_median]:
        B_prime_1 = 1
    elif r_mean_filtered[r_first] > r_mean_filtered[r_median] and \
         g_mean_filtered[g_first] > g_mean_filtered[g_median] and \
         b_mean_filtered[b_first] < b_mean_filtered[b_median]:
          B_prime_2 = 1
    # if B and B_prime:
    #     print("BB`")
           
    # print(R, R_prime, G, G_prime, B, B_prime)
    
    # R, G, B graph
    '''
    plt.subplot(411), plt.imshow(img)
    plt.subplot(412), plt.plot(r_mean, color='r'), plt.title(
        str(r_mean[r_max]) + ' ' + str(r_max) + ' ' + str(r_mean[r_min]) + ' ' + str(r_min))
    plt.axvline(x=r_max, color='r', linestyle='--')
    plt.axvline(x=r_min, color='b', linestyle='--')
    plt.axvline(x=r_median, color='g', linestyle='--')
    
    plt.plot(r_mean_filtered)
    plt.subplot(413), plt.plot(g_mean, color='g'), plt.title(
        str(g_mean[g_max]) + ' ' + str(g_max) + ' ' + str(g_mean[g_min]) + ' ' + str(g_min))
    plt.axvline(x=g_max, color='r', linestyle='--')
    plt.axvline(x=g_min, color='b', linestyle='--')
    plt.axvline(x=g_median, color='g', linestyle='--')
    
    plt.plot(g_mean_filtered)
    plt.subplot(414), plt.plot(b_mean, color='b'), plt.title(
        str(b_mean[b_max]) + ' ' + str(b_max) + ' ' + str(b_mean[b_min]) + ' ' + str(b_min))
    plt.axvline(x=b_max, color='r', linestyle='--')
    plt.axvline(x=b_min, color='b', linestyle='--')
    plt.axvline(x=b_median, color='g', linestyle='--')
    
    plt.plot(b_mean_filtered)
    plt.tight_layout()
    plt.show()
    '''

    rgb_max = [r_max, g_max, b_max]
    rgb_min = [r_min, g_min, b_min]
    rgb_pilot_1 = [R_1, R_prime_1, G_1, G_prime_1, B_1, B_prime_1]
    rgb_pilot_2 = [R_2, R_prime_2, G_2, G_prime_2, B_2, B_prime_2]
    rgb_start = [r_first, g_first, b_first]
    rgb_end = [r_second, g_second, b_second]

    # print(rgb_emax)
    # print(type(rgb_emax[0]))
    print(rgb_pilot_1)
    print(rgb_pilot_2)
    return rgb_start, rgb_end, rgb_max, rgb_min, rgb_pilot_1, rgb_pilot_2, pos

def GetColor(img, x_start, x_end, rgb_start, rgb_end, pos):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    red, green, blue = cv2.split(img)
 
    b_list = []
    g_list = []
    r_list = []
    c_list = []
    m_list = []
    ye_list = []
    for i in range(len(pos)):
        for idx in range(rgb_start[i], pos[i] + 1):
            y_list = np.argwhere(red[:, idx])
            if y_list.size > 0:
                y_start = y_list.min()
                y_end = y_list.max()
                
                if i == 0:
                    r_list.append(red[y_start:y_end + 1, idx].mean())
                elif i == 1:
                    g_list.append(green[y_start:y_end + 1, idx].mean())
                elif i == 2:    
                    b_list.append(blue[y_start:y_end + 1, idx].mean())
                    
        for idx in range(pos[i] + 1, rgb_end[i] + 1):
            y_list = np.argwhere(blue[:, idx])
            if y_list.size > 0:
                y_start = y_list.min()
                y_end = y_list.max()
                
                if i == 0:
                    c_list.append(red[y_start:y_end +1, idx].mean())
                elif i == 1:
                    m_list.append(green[y_start:y_end + 1, idx].mean())
                elif i == 2:
                    ye_list.append(blue[y_start:y_end + 1, idx].mean())
                    
    r_mean = np.array(r_list).mean()
    g_mean = np.array(g_list).mean()
    b_mean = np.array(b_list).mean()
    c_mean = np.array(c_list).mean()
    m_mean = np.array(m_list).mean()
    y_mean = np.array(ye_list).mean()
    
    return r_mean, g_mean, b_mean, c_mean, m_mean, y_mean

def estimating(r, g, b, channel_matrix, pseudo=False):
    color = np.array([[r], [g], [b]])

    # dot H
    # print(channel_matrix)
    if pseudo == False:
        color = np.dot(np.linalg.inv(channel_matrix), color)
    else:
        print(np.linalg.pinv(channel_matrix))
        color = np.dot(np.linalg.pinv(channel_matrix), color)
    # color = np.clip(color, 0 , 255)

    return color

def decoding(color):
    est_h, est_l, est_s = colorsys.rgb_to_hls(color[0],color[1],color[2])  
    # print('color :', est_h)
  
    code_hls_list = []
    for item in ColorSet:
        h, l, s = colorsys.rgb_to_hls(item.value[0], item.value[1], item.value[2])
        code_hls_list.append(h)

    code_hls_list = np.array(code_hls_list)
    # print(code_hls_list)

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

    # print("Est:{0}, Candidate:{1}".format(est_h, code_hls_list[minIdx]))
    return minIdx

def decoding2(rgb_color, cmy_color):
    est_h1, est_l1, est_s1 = colorsys.rgb_to_hls(rgb_color[0],rgb_color[1],rgb_color[2])  
    est_h2, est_l2, est_s2 = colorsys.rgb_to_hls(cmy_color[0],cmy_color[1],cmy_color[2])  
    # print('color :', est_h)
    
    est_h = (est_h1 + est_h2)/2
    code_hls_list = []
    for item in ColorSet:
        h, l, s = colorsys.rgb_to_hls(item.value[0], item.value[1], item.value[2])
        code_hls_list.append(h)

    code_hls_list = np.array(code_hls_list)
    # print(code_hls_list)

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

    # print("Est:{0}, Candidate:{1}".format(est_h, code_hls_list[minIdx]))
    return minIdx

def calculateBER(origin_data_list, rx_data_list, origin_bit_list):
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

    return fe
