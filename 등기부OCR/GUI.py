import re
import sys
from PyQt5.QtWidgets import *
from pandas import Series, DataFrame
import openpyxl

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        # 제목 입력
        self.setWindowTitle('엑셀 변환기')
        # 위젯의 크기 지정
        self.resize(400, 200)
        # 위젯의 위치 지정
        self.center()
        
        self.file_button = QPushButton('파일 탐색')
        self.file_button.clicked.connect(self.pushButtonClicked)
        self.label = QLabel()
        
        self.excel_button = QPushButton('엑셀 변환')
        self.excel_button.clicked.connect(self.convert_to_excel)
        
        layout = QVBoxLayout()
        layout.addWidget(self.file_button)
        layout.addWidget(self.excel_button)
        layout.addWidget(self.label)
        
        self.setLayout(layout)
        
        
    # 위젯의 위치를 화면의 가운데로 
    def center(self):
        # 사용하는 모니터 화면의 가운데 위치
        cen = QDesktopWidget().availableGeometry().center()
        # 창의 위치와 크기 정보를 가져온 후 화면의 중심으로 이동
        self.frameGeometry().moveCenter(cen)


    def pushButtonClicked(self):
        self.fname = QFileDialog.getOpenFileName(self)
        self.label.setText(self.fname[0])
    
    def cleanText(self, readData):
 
        #텍스트에 포함되어 있는 특수 문자 제거
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
        return text
    
    #동과 호수를 찾기위한 코드
    def donghosu(self, line):
        dong = None
        hosu = None
        line = line.split()
        
        for text in line:
            if text.startswith('제') and text.endswith('동') :
                    dong = text[1:3]
                    print(dong)
            if text.startswith('제') and text.endswith('호') :
                    hosu = text[1:4]
                    print(hosu)
        return dong, hosu
    
    def findhost(self, line):
        line = line.split()
        for text_ind in range(len(line)) :
            if line[text_ind] == '소유자' :
                host = line[text_ind+1]
                birth = line[text_ind+2]
                birth = birth[0:2]
        return host, birth
    
    # 주소, 건물내역, 대지권 비율 추출
    def extract_data(self, lines):
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
        
        return data_list
    
    def makeframe(self, result):
        df = DataFrame(result, columns=['동','호수','소유자', '주소', '건물내역', '대지권비율'])
        return df
    
    def makeexcel(self, df):
        df.to_excel('등기부등본.xlsx', # directory and file name to write

            sheet_name = 'Sheet1', 

            na_rep = 'NaN', 

            float_format = "%.2f", 

            header = True, 

            index = False, 

            startrow = 0, 

            startcol = 0, 

            #engine = 'xlsxwriter', 

            freeze_panes = (2, 0)

            )
    
    def showFile_1(self, filename):
        f = open(filename, 'r')
        lines = f.readlines()

        #갑구와 을구 사이를 구분하기 위한 카운터
        counter = 0
    
        result = []
        
        for line in lines :
            #특문제거
            line = self.cleanText(line)
            
            #동과 호수 식별
            if '집합건물' in line:
                print('집합건물 식별')
                d, h = self.donghosu(line)
                # 마지막으로 소유권이 이전된 사람을 식별하기 위한 코드   
            if '갑' in line and '구' in line and '소유권' in line :
                counter = 1
            if '소유자' in line and counter == 1:
                host, birth = self.findhost(line)
            if '을' in line and '구' in line and '소유권' in line :
                counter = 0
                # 최종 결과 데이터 생성
                data = [d, h, host +' '+ birth]
                result.append(data)
                print(result)
    
        # 주소, 건물내역, 대지권 비율 추출
        result2 = self.extract_data(lines)
    
        # 동, 호수, 소유자, 주소, 건물내역, 대지권 비율 합치기
        concat_result = list()
        for concat1, concat2 in zip(result, result2):
            concat_result.append(concat1+concat2)
        print(concat_result)
    
        f.close()
    
        # 데이터 프레임 생성
        deungi = self.makeframe(concat_result)
        print(deungi)
    
        # 데이터 프레임을 엑셀로 저장
        self.makeexcel(deungi)
    
    def convert_to_excel(self):
        print(self.fname[0])
        #file_name = self.fname[0].split('/')[-1]
        file_name = self.fname[0]
        print(file_name)
        self.showFile_1(self.fname[0])
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MyApp()
    gui.show()
    sys.exit(app.exec_())