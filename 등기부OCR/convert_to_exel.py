file = open('1동101_110호.txt', 'r')
lines = file.readlines()

# 전체 데이터에서 호수 분류를 위한 줄 번호 탐색
ho_idx = list()
idx = 0
for line in lines:
    if line.find('등기사항전부증명서') >= 0:
        ho_idx.append(idx)
    idx += 1

# 호수별 데이터 분류
ho = list()
for i in range(len(ho_idx)):
    if i == len(ho_idx)-1:
        ho.append(lines[ho_idx[i]:])
    else:
        ho.append(lines[ho_idx[i]:ho_idx[i+1]])

# 필요한 데이터 추출
data_list = list()
for i in range(len(ho)):
    data = list()
    # 주소 줄 번호
    address_start_idx = 0
    address_end_idx = 0
    # 표제부 줄 번호
    headline_idx = 0
    idx = 0
    for line in ho[i]:
        
        # 주소 줄 번호 탐색
        if line.find('갑 구') >= 0:
            address_start_idx = idx
        if line.find('을 구') >= 0:
            address_end_idx = idx
        # 표제부 줄 번호 탐색
        if line.find('표 제 부') >= 0:
            headline_idx = idx
        idx += 1
    
    # 가장 마지막 기록 추출
    address_idx = 0
    idx = address_start_idx
    for line in ho[i][address_start_idx:address_end_idx]:
        if line.find('전거') >= 0 or line.find('매매') >= 0:
            address_idx = idx
        idx += 1
        
    
    address = ''
    # 가장 마지막 기록이 '전거'일 경우
    if ho[i][address_idx].find('전거') >= 0:
        temp_idx = ho[i][address_idx-1].find('주소')
        address = ho[i][address_idx-1][temp_idx+3:-1]
        
        temp_idx = ho[i][address_idx].find('전거')
        address += ho[i][address_idx][temp_idx+2:-1]
    
    # 가장 마지막 기록이 '매매'일 경우
    if ho[i][address_idx].find('매매') >= 0:
        temp_idx = ho[i][address_idx].find('매매')
        address = ho[i][address_idx][temp_idx+3:-1]
        
        if ho[i][address_idx+1].find('호') >= 0:
            address += ho[i][address_idx+1][:-1]
        
    data.append(address)
            
    for line in ho[i][headline_idx:]:
        # 면적 추출
        if line.find('㎡') >= 0:
            data.append(line[line.find(')')+2:-1])

        # 소유대지권 추출
        if line.find('소유권대지권') >= 0:
            temp_str = line[line.find('소유권대지권'):]
            owned_land = temp_str.split(' ')[1:3]
            data.append(owned_land[0] + owned_land[1])
    
    data_list.append(data)
print(data_list)