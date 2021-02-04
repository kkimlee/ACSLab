import cv2
import numpy as np

# file path
# file = "C:\\Users\\user\\Documents\\Adobe\\Premiere Pro\13.0\\fr11_1.avi"
file = 'barcode_result.avi' # 쪼갤 영상의 경로

cap = cv2.VideoCapture(file)

if cap.isOpened() == False:
  print("Error opening video stream or file")

idx = 0

# directory path to save frames
# path = 'E:\\frames_105_4bit\\frs' # 프레임(스틸 이미지)를 저장할 경로, 'frames_영상파일 이름'
path = 'frame/frame'
while cap.isOpened():
    ret, frame = cap.read()
    
    # 이름 순으로 정렬하는 경우, fr13이 fr2보다 먼저 나오는 문제가 발생하므로
    # 빈 자리에 0을 매핑
    if ret==True:
        cv2.imshow('Frame', frame)
        if idx < 10:
            strIdx ="00"+str(idx)
        elif idx < 100:
            strIdx = "0"+str(idx)
        else:
            strIdx= str(idx)

        # frame = cv2.resize(frame, (1920, 1080))
        cv2.imwrite(path+strIdx+'.jpg',frame)
        idx += 1
    else:
        break

cap.release()
