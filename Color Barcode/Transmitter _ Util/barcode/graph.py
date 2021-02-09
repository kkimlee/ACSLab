import os
import cv2
import numpy as np
from scipy import signal
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


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
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
    print('B, G, R 성분 분화')
    rotated_b, rotated_g, rotated_r = cv2.split(img)

    # 90도 회전
    print('90도 회전')
    rotated_plt = cv2.merge([rotated_r.T, rotated_g.T, rotated_b.T])

    # 상하반전
    print('상하반전')
    rotated_plt = cv2.flip(rotated_plt, 0)

    # 최대 최소 고려
    print('최대 최소 고려')
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

    
    # 최대 최소 위치 비교
    r_first = compare(r_max, r_min, 0)
    r_second = compare(r_max, r_min, 1)
    g_first = compare(g_max, g_min, 0)
    g_second = compare(g_max, g_min, 1)
    b_first = compare(b_max, b_min, 0)
    b_second = compare(b_max, b_min, 1)
    
    print(r_first, r_second)
    print(g_first, g_second)
    print(b_first, b_second)
    # 중간값 
    r_median = np.argsort(r_mean_filtered)[len(r_mean_filtered)//2]
    g_median = np.argsort(g_mean_filtered)[len(g_mean_filtered)//2]
    b_median = np.argsort(b_mean_filtered)[len(b_mean_filtered)//2]
    
    R = 0
    R_prime = 0
    # R 찾기
    if r_mean_filtered[r_first] > r_mean_filtered[r_median] and \
       g_mean_filtered[g_first] < g_mean_filtered[g_median] and \
       b_mean_filtered[b_first] < b_mean_filtered[b_median]:
        R = 1
    if r_mean_filtered[r_second] < r_mean_filtered[r_median] and \
       g_mean_filtered[g_second] > g_mean_filtered[g_median] and \
       b_mean_filtered[b_second] > b_mean_filtered[b_median]:
        R_prime = 1
    if R and R_prime:
        print("RR`")
    
    G = 0
    G_prime = 0
    # G 찾기
    if r_mean_filtered[r_first] < r_mean_filtered[r_median] and \
       g_mean_filtered[g_first] > g_mean_filtered[g_median] and \
       b_mean_filtered[b_first] < b_mean_filtered[b_median]:
        G = 1
    if r_mean_filtered[r_second] > r_mean_filtered[r_median] and \
       g_mean_filtered[g_second] < g_mean_filtered[g_median] and \
       b_mean_filtered[b_second] > b_mean_filtered[b_median]:
        G_prime = 1
    if G and G_prime:
        print("GG`")
        
    B = 0
    B_prime = 0
    # B 찾기
    if r_mean_filtered[r_first] < r_mean_filtered[r_median] and \
       g_mean_filtered[g_first] < g_mean_filtered[g_median] and \
       b_mean_filtered[b_first] > b_mean_filtered[b_median]:
        B = 1
    if r_mean_filtered[r_second] > r_mean_filtered[r_median] and \
       g_mean_filtered[g_second] > g_mean_filtered[g_median] and \
       b_mean_filtered[b_second] < b_mean_filtered[b_median]:
        B_prime = 1
    if B and B_prime:
        print("BB`")
           
    print(R, R_prime, G, G_prime, B, B_prime)
    
    # R, G, B graph
    plt.subplot(411), plt.imshow(img)
    plt.subplot(412), plt.plot(r_mean, color='r'), plt.title(
        str(r_mean[r_max]) + ' ' + str(r_max) + ' ' + str(r_mean[r_min]) + ' ' + str(r_min))
    plt.axvline(x=r_max, color='r', linestyle='--')
    plt.axvline(x=r_min, color='b', linestyle='--')
    plt.axvline(x=r_median, color='g', linestyle='--')
    for x_ in r_emax:
        plt.axvline(x=x_, color='g', linewidth=1)
    for x_ in r_emin:
        plt.axvline(x=x_, color='k', linewidth=1)
    plt.plot(r_mean_filtered)
    plt.subplot(413), plt.plot(g_mean, color='g'), plt.title(
        str(g_mean[g_max]) + ' ' + str(g_max) + ' ' + str(g_mean[g_min]) + ' ' + str(g_min))
    plt.axvline(x=g_max, color='r', linestyle='--')
    plt.axvline(x=g_min, color='b', linestyle='--')
    plt.axvline(x=g_median, color='g', linestyle='--')
    for x_ in g_emax:
        plt.axvline(x=x_, color='g', linewidth=1)
    for x_ in g_emin:
        plt.axvline(x=x_, color='k', linewidth=1)
    plt.plot(g_mean_filtered)
    plt.subplot(414), plt.plot(b_mean, color='b'), plt.title(
        str(b_mean[b_max]) + ' ' + str(b_max) + ' ' + str(b_mean[b_min]) + ' ' + str(b_min))
    plt.axvline(x=b_max, color='r', linestyle='--')
    plt.axvline(x=b_min, color='b', linestyle='--')
    plt.axvline(x=b_median, color='g', linestyle='--')
    for x_ in b_emax:
        plt.axvline(x=x_, color='g', linewidth=1)
    for x_ in b_emin:
        plt.axvline(x=x_, color='k', linewidth=1)
    plt.plot(b_mean_filtered)
    plt.tight_layout()
    plt.show()
    

    pilot_red = []
    pilot_green = []
    pilot_blue = []

    rgb_emax = [r_emax, g_emax, b_emax]
    rgb_emin = [r_emin, g_emin, b_emin]
    rgb_max = [r_max, g_max, b_max]
    rgb_min = [r_min, g_min, b_min]
    rgb_pilot = [R, R_prime, G, G_prime, B, B_prime]

    print(rgb_emax)
    print(type(rgb_emax[0]))

    return rgb_emax, rgb_emin, rgb_max, rgb_min, rgb_pilot

def GetColor(img, x_start, x_end):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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

    r_mean = np.array(r_list).mean()
    g_mean = np.array(g_list).mean()
    b_mean = np.array(b_list).mean()

    return r_mean, g_mean, b_mean

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


frames = search('./', '.jpg')
frames.sort()

channel_matrix = [[0,0,0], [0,0,0], [0,0,0]]
for frame in frames:
    print(frame)
    img = cv2.imread(frame)
    cv2.imshow('barcode', img)
    cv2.waitKey(0)
    GetGraph(img, 0, 339, 0)
    GetPixelGraph(img, 0, 339, 0)
    r, g, b = GetColor(img, 0, 339)
    print(r, g, b)
    # rx_color = estimating(r, g, b, channel_matrix)

