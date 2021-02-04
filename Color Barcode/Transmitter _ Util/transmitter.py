import cv2
from enum import Enum

# openCV BGR
# RGB
class Color(Enum):
    RED = [255, 0, 0]
    REDCOM = [0, 255, 255]
    GREEN = [0, 255, 0]
    GREENCOM = [255, 0, 255]
    BLUE = [0, 0, 255]
    BLUECOM = [255, 255, 0]

# 3bit graycode
# class ColorSet(Enum):
#     C000 = [255,0,0] 
#     C001 = [255,191,0] 
#     C011 = [127,255,0] 
#     C010 = [0,255,63] 
#     C110 = [0,255,255] 
#     C111 = [0,63,255]
#     C101 = [127,0,255]
#     C100 = [255,0,191]
#
# # 3bit data frame
# # 27 data + 27 comp
# bit_dict = {
#     "000": ColorSet.C000.value,
#     "001": ColorSet.C001.value,
#     "011": ColorSet.C011.value,
#     "010": ColorSet.C010.value,
#     "110": ColorSet.C110.value,
#     "111": ColorSet.C111.value,
#     "101": ColorSet.C101.value,
#     "100": ColorSet.C100.value
# }

# 4bit graycode
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

BIT_SIZE = 4
CODE_SIZE = 16

# pilot frame
# RR`GG`BB`

# 일단 송신 데이터 리스트 생성
bit_list = []
for code in ColorSet:
    bit_list.append(code.value)

# data_list = [ 2, 4, 3, 6, 0, 5 ] * 4
# data_list = data_list + [ 1, 7, 3 ]

# 4bit
data_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
data_list = data_list + [ 11, 12, 13, 14, 15 ,0, 1]

# 3bit
# data_list = [1, 2, 3, 4, 5, 6, 7] * 3
# data_list = data_list + [0, 1, 2, 3, 4, 5]

# data frame making
# col+comp_col
df_list = []
for idx in range(len(data_list)):
    df_list.append(bit_list[data_list[idx]])
    df_list.append(bit_list[(data_list[idx]+int(CODE_SIZE/2))%CODE_SIZE])

print(df_list)

img = cv2.imread("./display.jpg",cv2.IMREAD_COLOR)
img = cv2.resize(img, dsize=(1920,1080), interpolation=cv2.INTER_AREA)
height, width, channel = img.shape

print(height)
print(width)
print(channel)

# 파일럿 RR`GG`BB`
pilot_list = [Color.RED.value,Color.REDCOM.value,Color.GREEN.value,Color.GREENCOM.value,Color.BLUE.value,Color.BLUECOM.value]

# 파일럿과 데이터 합침
df_list = pilot_list+df_list

barHeight = 50
barWidth = width

cv2.imshow("img",img)
cv2.waitKey(0)

# 영상의 이름, 코덱, fps, 크기 설정
out = cv2.VideoWriter('4bit.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (width,height))

# 10 sec, 60fps, pilot+data
for frame in range(1200):
    r, g, b = df_list[frame%60]
    img[:barHeight,:barWidth,:] = [b, g, r]
    cv2.imwrite('./still/fr'+str(frame)+'.jpg',img)
    out.write(img)
out.release()