import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pywt

# 경로내에 입력받은 확장자형식의 파일 탐색
def search(dirname, extension):
    # 파일을 저장할 리스트
    file_list = []
    # 파일을 탐색할 경로 저장
    filenames = os.listdir(dirname)
    
    for filename in filenames:
        # 경로내에 있는 모든 파일 탐색
        full_filename = os.path.join(dirname, filename)
        # 탐색된 파일에서 확장자만 추출
        ext = os.path.splitext(full_filename)[-1]
        # 추출된 확장자가 지정한 확장자와 같은 경우
        if ext == extension:
            # 리스트에 저장할 변수에 파일명 추가
            file_list.append(full_filename)
            print(full_filename)

    return file_list

# 현재 폴더 내에 csv 형식의 파일 탐색
data_list = search('./', '.csv')

# 크랙게이지 데이터가 저장되어있는 파일명을 저장하기 위한 리스트
crackgauge_data_list = []
# 구조물 경사계 데이터가 저장되어있는 파일명을 저장하기 위한 리스트
tiltmeter_data_list = []

# 크랙게이지 데이터와 구조물 경사계 데이터는 대합실에 따라 저장되어 있음
# 크랙게이지는 C-1~5, 구조물 경사계는 T-1~5로 파일명이 저장되어 있음
# csv 파일들을 순서대로 탐색
for file in data_list:
    # 파일명이 C로 시작하는 경우 
    if (file[2] == 'C'):
        # 크랙게이지 데이터 파일 리스트에 추가
        crackgauge_data_list.append(file)
    # 파일명이 T로 시작하는 경우 
    elif (file[2] == 'T'):
        # 구조물 경사계 데이터 파일 리스트에 추가
        tiltmeter_data_list.append(file)




# 크랙게이지 데이터를 저장할 리스트
crackgauge_data = []
# 크랙게이지 파일 리스트에서 파일명을 하나씩 읽어옴
for crackgauge_file in crackgauge_data_list:
    # 파일명을 이용해 데이터를 읽어옴
    data = pd.read_csv(crackgauge_file,
                       names=['#', 'datetime', 'value', 'displacement(mm)', 'remark'])
    # 크랙게이지 데이터 리스트에 읽어온 데이터 추가
    crackgauge_data.append(data)

# 구조물 경사계 데이터를 저장할 리스트
tiltmeter_data = []
# 구조물 경사계 파일 리스트에서 파일명을 하나씩 읽어옴    
for tiltmeter_file in tiltmeter_data_list:
    # 파일명을 이용해 데이터를 읽어옴
    data = pd.read_csv(tiltmeter_file,
                       names=['#', 'datetime', 'x_value', 'y_value', 'x_displacement(mm)', 'y_displacement(mm)', 'remark'])
    # 구조물 경사계 리스트에 읽어온 데이터 추가
    tiltmeter_data.append(data)
                         
# 크랙게이지 데이터와 구조물 경사계 데이터를 합쳐서 저장할 리스트
merge_data = []
for i in range(len(crackgauge_data)):
    # 크랙게이지 데이터와 구조물 경사계 데이터를 합침
    data = pd.merge(crackgauge_data[i], tiltmeter_data[i])
    # 합쳐진 데이터를 리스트에 추가  
    merge_data.append(data)


for i in range(len(merge_data)):
   fig = plt.figure()
   
   # 그래프의 제목
   title1 = str(i+1) + 'area crackgauge_displacement(mm), initial value : ' + str(merge_data[i]['value'][0])
   title2 = str(i+1) + 'area tiltmeter_x_displacement(mm), initial value : ' + str(merge_data[i]['x_value'][0])
   title3 = str(i+1) + 'area tiltmeter_y_displacement(mm), initial value : ' + str(merge_data[i]['y_value'][0])
   
   # 그래프에서 표현할 범위 조절
   displacement_min = float(max(merge_data[i]['displacement(mm)'][1:]))
   displacement_max = float(min(merge_data[i]['displacement(mm)'][1:]))
   
   # 크랙게이지 변위 시각화
   ax1 = fig.add_subplot(3, 1, 1)  
   ax1.set_title(title1)
   ax1.plot(merge_data[i]['displacement(mm)'])
   
   # 구조물 경사계 x축 변위 시각화
   ax2 = fig.add_subplot(3, 1, 2)
   ax2.set_title(title2)
   ax2.plot(merge_data[i]['x_displacement(mm)'])
   
   # 구조물 경사계 y축 변위 시각화
   ax3 = fig.add_subplot(3, 1, 3)
   ax3.set_title(title3)
   ax3.plot(merge_data[i]['y_displacement(mm)'])
   
   plt.tight_layout()
   plt.show()

for i in range(len(merge_data)):
    # 측정값에 "," 문자 제거
    merge_data[i]['value'] = merge_data[i]['value'].str.replace(',', '')
    merge_data[i]['x_value'] = merge_data[i]['x_value'].str.replace(',','')
    merge_data[i]['y_value'] = merge_data[i]['y_value'].str.replace(',','')
    
    # 측정값을 float형으로 변환                                                                    
    merge_data[i]['value'] = merge_data[i]['value'].astype(float)
    merge_data[i]['x_value'] = merge_data[i]['x_value'].astype(float)
    merge_data[i]['y_value'] = merge_data[i]['y_value'].astype(float)
    
    # 제목 설정
    fig = plt.figure()
    title = str(i+1) + 'area heatmap'
    plt.title(title)
    
    # 상관관계 분석을 위해 히트맵 사용
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.heatmap(merge_data[i].corr(), annot=True, fmt='.2f')
    
    plt.show()
'''
for i in range(len(merge_data)):
    merge_data[i]['abs_displacement(mm)'] = abs(merge_data[i]['displacement(mm)'])
    merge_data[i]['abs_x_displacement(mm)'] = abs(merge_data[i]['x_displacement(mm)'])
    merge_data[i]['abs_y_displacement(mm)'] = abs(merge_data[i]['y_displacement(mm)'])
    
    merge_data[i]['xy_displacement_sub_abs'] = abs(merge_data[i]['x_displacement(mm)'] - merge_data[i]['y_displacement(mm)'])
    merge_data[i]['xy_displacement_avg'] = (merge_data[i]['x_displacement(mm)'] + merge_data[i]['y_displacement(mm)'])/2
    
    merge_data[i]['xy_displacement_ratio'] = abs(merge_data[i]['abs_x_displacement(mm)']/merge_data[i]['abs_y_displacement(mm)'])
    
for i in range(len(merge_data)):
    fig = plt.figure()
    title = str(i+1) + 'area heatmap'
    plt.title(title)
    
    ax1 = fig.add_subplot(1, 1, 1)
    
    ax1 = sns.heatmap(merge_data[i].corr(), annot=True, fmt='.2f')
    plt.show()
'''
merge_data2 = pd.DataFrame()

merge_data2['1_area_displacement(mm)'] = merge_data[0]['displacement(mm)']
merge_data2['1_area_x_dispalcement(mm)'] = merge_data[0]['x_displacement(mm)']
merge_data2['1_area_y_dispalcement(mm)'] = merge_data[0]['y_displacement(mm)']
merge_data2['2_area_displacement(mm)'] = merge_data[1]['displacement(mm)']
merge_data2['2_area_x_dispalcement(mm)'] = merge_data[1]['x_displacement(mm)']
merge_data2['2_area_y_dispalcement(mm)'] = merge_data[1]['y_displacement(mm)']
merge_data2['4_area_displacement(mm)'] = merge_data[3]['displacement(mm)']
merge_data2['4_area_x_dispalcement(mm)'] = merge_data[3]['x_displacement(mm)']
merge_data2['4_area_y_dispalcement(mm)'] = merge_data[3]['y_displacement(mm)']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax = sns.heatmap(merge_data2.corr(), annot=True, fmt='.2f')
plt.show()

'''
1, 2, 4 데이터 분석
0~380까지는 변위가 0
381~462는 변위 변화
463~603은 다시 0
'''
merge_data3 = []
merge_data3.append(merge_data[0])
merge_data3.append(merge_data[1])
merge_data3.append(merge_data[3])

for i in range(len(merge_data3)):
    # 균열 변화 전 데이터
    before_data = merge_data3[i][:381]
    # 균열 변화 후 데이터
    after_data = merge_data3[i][463:]

    # 제목 설정
    area = ['1', '2', '4']
    title = 'area' + area[i] + 'before-after wavelet'
    
    fig = plt.figure()
    plt.title(title, position=(0.5, 1.1))
    plt.axis('off')

    # 균열 변화 전 x축 변위 웨이블릿 변환
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('before x displacement')
    coef, freqs = pywt.cwt(before_data['x_displacement(mm)'], np.arange(1, len(before_data)), 'morl')
    ax1.imshow(coef, cmap='gray')

    # 균열 변화 전 y축 변위 웨이블릿 변환
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('before y displacement')
    coef, freqs = pywt.cwt(before_data['y_displacement(mm)'], np.arange(1, len(before_data)), 'morl')
    ax2.imshow(coef, cmap='gray')

    # 균열 변화 후 x축 변위 웨이블릿 변환
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('after x displacement')
    coef, freqs = pywt.cwt(after_data['x_displacement(mm)'], np.arange(1, len(after_data)), 'morl')
    ax3.imshow(coef, cmap='gray')

    # 균열 변화 후 y축 변위 웨이블릿 변환
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('after y displacement')
    coef, freqs = pywt.cwt(after_data['y_displacement(mm)'], np.arange(1, len(after_data)), 'morl')
    ax4.imshow(coef, cmap='gray')

    plt.tight_layout()
    plt.show()

    # 균열 변화 전 300step ~ 200step 데이터
    section1 = merge_data3[i][81:181]
    # 균열 변화 전 200step ~  100step 데이터
    section2 = merge_data3[i][181:281]
    # 균열 변화 전 100step ~  0step 데이터
    section3 = merge_data3[i][281:381]
    # 균열 변화 후 0step ~ 100step 까지
    section4 = merge_data3[i][463:563]

    # 제목 설정
    title2 = 'area' + area[i] + ' 100step wavelet'

    # 각 구간별 x축 변위 웨이블릿 변환
    fig = plt.figure()
    plt.title(title2, position=(0.5, 1.1))
    plt.axis('off')
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('section1 x displacement')
    coef, freqs = pywt.cwt(section1['x_displacement(mm)'], np.arange(1, len(section1)), 'morl')
    ax1.imshow(coef, cmap='gray')
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('section2 x displacement')
    coef, freqs = pywt.cwt(section2['x_displacement(mm)'], np.arange(1, len(section2)), 'morl')
    ax2.imshow(coef, cmap='gray')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('section3 x displacement')
    coef, freqs = pywt.cwt(section3['x_displacement(mm)'], np.arange(1, len(section3)), 'morl')
    ax3.imshow(coef, cmap='gray')
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('section4 x displacement')
    coef, freqs = pywt.cwt(section4['x_displacement(mm)'], np.arange(1, len(section4)), 'morl')
    ax4.imshow(coef, cmap='gray')
    
    plt.tight_layout()
    plt.show()

    # 각 구간별 y축 변위 웨이블릿 변환
    fig = plt.figure()
    plt.title(title2, position=(0.5, 1.1))
    plt.axis('off')
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('section1 y displacement')
    coef, freqs = pywt.cwt(section1['y_displacement(mm)'], np.arange(1, len(section1)), 'morl')
    ax1.imshow(coef, cmap='gray')
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('section2 y displacement')
    coef, freqs = pywt.cwt(section2['y_displacement(mm)'], np.arange(1, len(section2)), 'morl')
    ax2.imshow(coef, cmap='gray')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('section3 y displacement')
    coef, freqs = pywt.cwt(section3['y_displacement(mm)'], np.arange(1, len(section3)), 'morl')
    ax3.imshow(coef, cmap='gray')
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('section4 y displacement')
    coef, freqs = pywt.cwt(section4['y_displacement(mm)'], np.arange(1, len(section4)), 'morl')
    ax4.imshow(coef, cmap='gray')
    
    plt.tight_layout()
    plt.show()

for i in range(len(merge_data3)):
    before_data = merge_data3[i][:381]
    after_data = merge_data3[i][463:] 
    area = ['1', '2', '4']
    
    title = 'area' + area[i] + 'before-after fft-phase'
    
    fig = plt.figure()
    plt.title(title, position=(0.5, 1.1))
    plt.axis('off')
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('before x displacement')
    pha = np.angle(np.fft.fft(before_data['x_displacement(mm)']))
    ax1.plot(pha)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('before y displacement')
    pha = np.angle(np.fft.fft(before_data['y_displacement(mm)']))
    ax2.plot(pha)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('after x displacement')
    pha = np.angle(np.fft.fft(after_data['x_displacement(mm)']))
    ax3.plot(pha)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('after y displacement')
    pha = np.angle(np.fft.fft(after_data['y_displacement(mm)']))
    ax4.plot(pha)

    plt.tight_layout()
    plt.show()
    
    section1 = merge_data3[i][81:181]
    section2 = merge_data3[i][181:281]
    section3 = merge_data3[i][281:381]
    section4 = merge_data3[i][463:563]
    
    title2 = 'area' + area[i] + ' 100step fft-phase'
    
    fig = plt.figure()
    plt.title(title2, position=(0.5, 1.1))
    plt.axis('off')
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('section1 x displacement')
    pha = np.angle(np.fft.fft(section1['x_displacement(mm)']))
    ax1.plot(pha)
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('section2 x displacement')
    pha = np.angle(np.fft.fft(section2['x_displacement(mm)']))
    ax2.plot(pha)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('section3 x displacement')
    pha = np.angle(np.fft.fft(section3['x_displacement(mm)']))
    ax3.plot(pha)
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('section4 x displacement')
    pha = np.angle(np.fft.fft(section4['x_displacement(mm)']))
    ax4.plot(pha)
    
    plt.tight_layout()
    plt.show()
    
    fig = plt.figure()
    plt.title(title2, position=(0.5, 1.1))
    plt.axis('off')
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('section1 y displacement')
    pha = np.angle(np.fft.fft(section1['y_displacement(mm)']))
    ax1.plot(pha)
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('section2 y displacement')
    pha = np.angle(np.fft.fft(section2['y_displacement(mm)']))
    ax2.plot(pha)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('section3 y displacement')
    pha = np.angle(np.fft.fft(section3['y_displacement(mm)']))
    ax3.plot(pha)
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('section4 y displacement')
    pha = np.angle(np.fft.fft(section4['y_displacement(mm)']))
    ax4.plot(pha)
    
    plt.tight_layout()
    plt.show()
