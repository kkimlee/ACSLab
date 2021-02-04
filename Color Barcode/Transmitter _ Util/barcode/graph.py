import os
import cv2
import numpy as np
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
    r, g, b = GetColor(img, 0, 339)
    print(r, g, b)
    # rx_color = estimating(r, g, b, channel_matrix)

