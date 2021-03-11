import os
import re
import sys
import openpyxl
from pandas import Series, DataFrame

from PyQt5.QtWidgets import *
import win32com.client as win32

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        # 제목 입력
        self.setWindowTitle('데이터 추출기')
        # 위젯의 크기 지정
        self.resize(400, 200)
        # 위젯의 위치 지정
        self.center()
        
        # 파일 탐색 버튼
        self.file_button = QPushButton('파일 탐색')
        self.file_button.clicked.connect(self.pushButtonClicked)
        self.label = QLabel()
        
        # 엑셀 변환 버튼
        self.excel_button = QPushButton('엑셀 변환')
        self.excel_button.clicked.connect(self.convert_to_excel)
        
        # 한글 변환 버튼
        self.hangul_button = QPushButton('한글 변환')
        self.hangul_button.clicked.connect(self.convert_to_hangul)
        
        # 한글 양식 1번
        self.hangul_form_1 = QPushButton('한글 양식 1')
        self.hangul_form_1.clicked.connect(self.convert_to_hangul_form_1)
        
        # 한글 양식 2번
        self.hangul_form_2 = QPushButton('한글 양식 2')
        self.hangul_form_2.clicked.connect(self.convert_to_hangul_form_2)
        
        # 한글 양식 3번
        self.hangul_form_3 = QPushButton('한글 양식 3')
        self.hangul_form_3.clicked.connect(self.convert_to_hangul_form_3)
        
        # 한글 양식 4번
        self.hangul_form_4 = QPushButton('한글 양식 4')
        self.hangul_form_4.clicked.connect(self.convert_to_hangul_form_4)
        
        # 레이아웃
        layout = QVBoxLayout()
        layout.addWidget(self.file_button)
        layout.addWidget(self.excel_button)
        layout.addWidget(self.hangul_button)
        layout.addWidget(self.hangul_form_1)
        layout.addWidget(self.hangul_form_2)
        layout.addWidget(self.hangul_form_3)
        layout.addWidget(self.hangul_form_4)
        layout.addWidget(self.label)
        
        self.setLayout(layout)
        
        self.fname = ''
        self.form_name = ''
    # 위젯의 위치를 화면의 가운데로 
    def center(self):
        # 사용하는 모니터 화면의 가운데 위치
        cen = QDesktopWidget().availableGeometry().center()
        # 창의 위치와 크기 정보를 가져온 후 화면의 중심으로 이동
        self.frameGeometry().moveCenter(cen)


    # 파일 탐색 버튼 클릭시
    def pushButtonClicked(self):
        self.fname = QFileDialog.getOpenFileName(self)
        self.label.setText(self.fname[0])
    
    # 데이터 파일 선택 확인
    def checkDataFile(self):
        if self.fname == '' or self.fname[0] == '':
            reply = QMessageBox.question(self, 'Message', '파일을 선택해 주세요', QMessageBox.Yes, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.pushButtonClicked()
                
                if self.fname == '' or self.fname[0] == '':
                    return False
                elif self.fname[0].split('.')[1] != 'txt':
                    reply = QMessageBox.question(self, 'Message', '잘못된 형식의 파일입니다', QMessageBox.Yes)
                    self.fname = ''
                    return False
                else:
                    return True
        elif self.fname[0].split('.')[1] != 'txt':
            reply = QMessageBox.question(self, 'Message', '잘못된 형식의 파일입니다', QMessageBox.Yes)
            self.fname = ''
            return False
        else:
            return True
        
    # 양식 파일 선택 확인
    def checkFormFile(self):
        if self.form_name == '' or self.form_name[0] == '':
            reply = QMessageBox.question(self, 'Message', '양식 파일을 선택해 주세요', QMessageBox.Yes, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.form_name = QFileDialog.getOpenFileName(self)
        
                if self.form_name == '' or self.form_name[0] == '':
                    return False
                elif self.form_name[0].split('.')[1] != 'hwp':
                    reply = QMessageBox.question(self, 'Message', '잘못된 형식의 양식 파일입니다.', QMessageBox.Yes)
                    self.form_name = ''
                else:
                    return True
        elif self.form_name[0].split('.')[1] != 'hwp':
            reply = QMessageBox.question(self, 'Message', '잘못된 형식의 양식 파일입니다.', QMessageBox.Yes)
            self.form_name = ''
            return False
        else:
            return True
    # 특수 문자 제거 
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
                    # print(dong)
            if text.startswith('제') and text.endswith('호') :
                    hosu = text[1:4]
                    # print(hosu)
        return dong, hosu
    
    # 소유자 찾기
    def findhost(self, line):
        line = line.split()
        for text_ind in range(len(line)) :
            if line[text_ind] == '소유자' :
                try :
                    host = line[text_ind+1]
                    birth = line[text_ind+2]
                    birth = birth[0:6]
                    return host, birth
                except :
                    host = line[text_ind+1]
                    host = ''.join(host.split())
                    return host
                
    def findshare(self, line):
        line = line.split()
        
        host = line[-2]
        birth = line[-1]
        birth = birth[0:6]
        return host + ' ' +birth
    
    
    def remove_data(self, line):
        if line.find('거래가액') <  0 and line.find('을 구') < 0 and \
            line.find('부동산등기법') < 0 and line.find('말소사항') < 0 and \
            line.find('*******') < 0  and line.find('청구금액') < 0 and \
            line.find('금지사항') < 0 and line.find('지분') < 0 and \
            line.find('성명') < 0 and line.find('주소') < 0:
            return True
        else:
            return False
    
    # 주소, 건물내역, 대지권 비율 추출
    def extract_data(self, lines):
        remove = list()
        idx = 0
        for line in lines:
            if line.find('발행번호') >= 0 or line.find('집합건물') >= 0 or \
                line.find('/') >= 0 or line.find('순위번호') >= 0 or \
                len(line)==2 or line.find('열 람 용') >= 0 or \
                line.find('열람일시') >= 0:
                remove.append(idx)
            idx += 1
    
        for i in range(len(remove)):
            del lines[remove[i]-i]
        
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
                if line.find('전거') >= 0 or line.find('매매') >= 0 or \
                   line.find('증여') >= 0 or line.find('상속') >= 0 or \
                   line.find('낙찰') >= 0:
                   
                    if line.find('매매예약') < 0:
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
                if ho[i][address_idx-1].find('공유자') >= 0:
                    address = ho[i][address_idx+2]
                
                    idx = 2
                    while(True):
                        idx += 1
                        if self.remove_data(ho[i][address_idx+idx]):
                            address += ho[i][address_idx+idx][:-1]
                        else:
                            break
                        
                if ho[i][address_idx-1].find('소유자') >= 0:
                    owner_idx = ho[i][address_idx-1].find('소유자')
                    owner = ho[i][address_idx-1][owner_idx:].split()[1]
                    if (len(owner) > 10):
                        print(owner)
                        idx = 0
                        while(True):
                            idx += 1
                            if self.remove_data(ho[i][address_idx+idx]):
                                address += ho[i][address_idx+idx][:-1]
                            else:
                                break
                    else:
                        temp_idx = ho[i][address_idx].find('매매')
                        address = ho[i][address_idx][temp_idx+3:-1]
                    
                        idx = 0         
                        while(True):
                            idx += 1
                            if self.remove_data(ho[i][address_idx+idx]):
                                address += ho[i][address_idx+idx][:-1]
                            else:
                                break

            # 가장 마지막 기록이 '증여'일 경우
            if ho[i][address_idx].find('증여') >= 0:
                if ho[i][address_idx-1].find('공유자') >= 0:
                    if ho[i][address_idx-1].find('지분') >= 0:
                        address = ho[i][address_idx+1]
                        
                        idx = 1
                        while(True):
                            idx += 1
                            if self.remove_data(ho[i][address_idx+idx]):
                                address += ho[i][address_idx+idx][:-1]
                            else:
                                break
                    else:    
                        address = ho[i][address_idx+2]
                    
                        idx = 2
                        while(True):
                            idx += 1
                            if self.remove_data(ho[i][address_idx+idx]):
                                address += ho[i][address_idx+idx][:-1]
                            else:
                                break
                else:
                    temp_idx = ho[i][address_idx].find('증여')
                    address = ho[i][address_idx][temp_idx+3:-1]
                
                    idx = 0 
                    while(True):
                        idx += 1
                        if self.remove_data(ho[i][address_idx+idx]):
                            address += ho[i][address_idx+idx][:-1]
                        else:
                            break
            
            # 가장 마지막 기록이 '상속'일 경우
            if ho[i][address_idx].find('상속') >= 0:
                if ho[i][address_idx-1].find('협의분할에') >= 0:
                    temp_idx = ho[i][address_idx-1].find('협의분할에')
                    address = ho[i][address_idx-1][temp_idx+6:-1]
                
                if ho[i][address_idx-1].find('협의분할') >= 0:
                    temp_idx = ho[i][address_idx-1].find('협의분할')
                    address = ho[i][address_idx-1][temp_idx+5:-1]
                    
                temp_idx = ho[i][address_idx].find('상속')
                address += ho[i][address_idx][temp_idx+2:-1]
                
                idx = 0 
                while(True):
                    idx += 1
                    if self.remove_data(ho[i][address_idx+idx]):
                        address += ho[i][address_idx+idx][:-1]
                    else:
                        break
            
            # 가장 마지막 기록이 '낙찰'일 경우
            if ho[i][address_idx].find('낙찰') >= 0:
                temp_idx = ho[i][address_idx-1].find('임의경매로')
                address = ho[i][address_idx-1][temp_idx+6:-1]
                
                temp_idx = ho[i][address_idx].find('낙찰')
                address += ho[i][address_idx][temp_idx+2:-1]
                
                idx = 0 
                while(True):
                    idx += 1
                    if self.remove_data(ho[i][address_idx+idx]):
                        address += ho[i][address_idx+idx][:-1]
                    else:
                        break
                    
            data.append(address)
            
            
            
            for line in ho[i][headline_idx:address_start_idx]:
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
    
    # 데이터 프레임형태로 변
    def makeframe(self, result):
        df = DataFrame(result, columns=['동','호수','소유자','소유자수', '주소', '건물내역', '대지권비율'])
        return df
    
    # 엑셀 파일 생성
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
    
    # 데이터 추출 
    def showFile_1(self, filename):
        f = open(filename, 'r')
        try:
            lines = f.readlines()
        except:
            f = open(filename, 'r', encoding='utf-8')
            lines = f.readlines()

        #갑구와 을구 사이를 구분하기 위한 카운터
        counter = 0

        result = []
        
        # 동, 호수, 소유자 추출
        for line in lines :
            #특문제거
            line = self.cleanText(line)
            #동과 호수 식별
            if '집합건물' in line:
                d, h = self.donghosu(line)
                # 마지막으로 소유권이 이전된 사람을 식별하기 위한 코드   
            if '갑' in line and '구' in line and '소유권' in line :
                counter = 1
            
            if '소유자' in line:
                counter = 1
                host = self.findhost(line)
            elif '공유자' in line:
                counter = 2
                host = []
            
                
            if counter == 2 and '지분' in line :
                jiboon = 1
                
            if counter == 2 and len(line.split()) == 2 and jiboon == 1:
                share = self.findshare(line)
                host.append(share)
                jiboon = 0
            elif counter == 2 and '이전' in line and '매매' in line and jiboon == 1:
                share = self.findshare(line)
                host.append(share)
                jiboon = 0
            elif counter == 2 and '증여' in line:
                share = self.findshare(line)
                host.append(share)
            
            if '을' in line and '구' in line and '소유권' in line :
                counter = 0
                jiboon = 0
                # 결과가 공유자일 시 리스트를 텍스트로 변환
                if str(type(host)) == "<class 'list'>":
                    num = len(host)
                    host = ' '.join(host)
                    print('리스트 형태 호스트 결과 출력 :',host)
                    
                else:
                    if str(type(host)) == "<class 'str'>":
                        num = 1
                        host = ''.join(host)
                    else:
                        num = len(host)/2                
                        host = ' '.join(host)
                        
                
                # 최종 결과 데이터 생성
                data = [d, h, host, num]
                result.append(data)

    
    
        # 주소, 건물내역, 대지권 비율 추출
        result2 = self.extract_data(lines)
    
        # 동, 호수, 소유자, 주소, 건물내역, 대지권 비율 합치기
        concat_result = list()
        for concat1, concat2 in zip(result, result2):
            concat_result.append(concat1+concat2)
        # print(concat_result)
    
        f.close()
    
        # 데이터 프레임 생성
        deungi = self.makeframe(concat_result)
        # print(deungi)
        
        
        return deungi
    
    # 엑셀로 변환 버튼 클릭
    def convert_to_excel(self):
        if self.checkDataFile():
            deungi = self.showFile_1(self.fname[0])
            self.makeexcel(deungi)
        
    # 한글로 변환 버튼 클릭
    def convert_to_hangul(self):
        if self.checkDataFile():
        
            deungi = self.showFile_1(self.fname[0])
            path = os.getcwd()
            owner = list(deungi['소유자'])
            address = list(deungi['주소'])
    
            hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
            hwp.XHwpWindows.Item(0).Visible = True
            for i in range(len(owner)):
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                string = '소유자 : ' + owner[i] + ' 주소 : ' + address[i]
                hwp.HParameterSet.HInsertText.Text = string
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                if i < len(owner)-1:
                    hwp.HAction.Run("BreakPage");
                    
                hwp.SaveAs(path + '/등기부등본.hwp')
                hwp.Quit() 
        # 표 만들기
        '''
        hwp.HAction.GetDefault("TableCreate", hwp.HParameterSet.HTableCreation.HSet)
        hwp.HParameterSet.HTableCreation.Rows = 1
        hwp.HParameterSet.HTableCreation.Cols = 2
        hwp.HParameterSet.HTableCreation.WidthValue = hwp.MiliToHwpUnit(0.0)
        hwp.HParameterSet.HTableCreation.HeightValue = hwp.MiliToHwpUnit(0.0)
        hwp.HParameterSet.HTableCreation.CreateItemArray("ColWidth", 2)
        hwp.HParameterSet.HTableCreation.CreateItemArray("RowHeight", 1)
        hwp.HParameterSet.HTableCreation.TableProperties.Width = 41954
        hwp.HAction.Execute("TableCreate", hwp.HParameterSet.HTableCreation.HSet)
        hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
        hwp.HParameterSet.HInsertText.Text = "소유자"
        hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
        hwp.HAction.Run("TableRightCellAppend")
        hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
        hwp.HParameterSet.HInsertText.Text = "주소"
        hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
            
        for i in range(len(owner)):
            hwp.HAction.Run("TableAppendRow")
            hwp.HAction.Run("MoveLeft");
            hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
            hwp.HParameterSet.HInsertText.Text = owner[i]
            hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
            hwp.HAction.Run("TableRightCellAppend")
            hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
            hwp.HParameterSet.HInsertText.Text = address[i]
            hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
        '''
          
             
    # 한글양식으로 변환 버튼 클릭
    def convert_to_hangul_form_1(self):
        if self.checkDataFile():
            self.form_name = QFileDialog.getOpenFileName(self)
            print('양식 파일 선')
            
            if self.checkFormFile():
                path = os.getcwd()
                deungi = self.showFile_1(self.fname[0])
                owner = list(deungi['소유자'])
                address = list(deungi['주소'])
        
                hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
                hwp.XHwpWindows.Item(0).Visible = True
                # hwp.Open(path + '/서면결의서.hwp')
                hwp.Open(self.form_name[0])
        
                hwp.HAction.Run("MoveTopLevelBegin")
                hwp.HAction.Run("SelectAll")
                hwp.HAction.Run("Copy")
                hwp.HAction.Run("MoveLineBegin")
                hwp.HAction.Run("MoveDown")
                hwp.HAction.Run("MoveDown")
                hwp.HAction.Run("MoveDown")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HParameterSet.HInsertText.Text = owner[0].split()[0]
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HParameterSet.HInsertText.Text = owner[0].split()[1]
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HParameterSet.HInsertText.Text = address[0]
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                
                for i in range(len(deungi) - 1):
                    hwp.HAction.Run("MoveDown")
                    hwp.HAction.Run("MoveViewDown");
                    hwp.HAction.Run("MoveViewDown");
                    hwp.HAction.Run("BreakPage")
                    hwp.HAction.GetDefault("Paste", hwp.HParameterSet.HSelectionOpt.HSet)
                    hwp.HAction.Execute("Paste", hwp.HParameterSet.HSelectionOpt.HSet)
                    hwp.HAction.Run("MovePageUp")
                    hwp.HAction.Run("MoveDown")
                    hwp.HAction.Run("MoveDown")
                    hwp.HAction.Run("MoveDown")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HParameterSet.HInsertText.Text = owner[i+1].split()[0]
                    hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HParameterSet.HInsertText.Text = owner[i+1].split()[1]
                    hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HParameterSet.HInsertText.Text = address[i+1]
                    hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
            
            # hwp.Quit()
            self.form_name = ''
        
        # 한글양식으로 변환 버튼 클릭
    def convert_to_hangul_form_2(self):
        if self.checkDataFile():
            self.form_name = QFileDialog.getOpenFileName(self)
            
            if self.checkFormFile():
                path = os.getcwd()
                deungi = self.showFile_1(self.fname[0])
                owner = list(deungi['소유자'])
                address = list(deungi['주소'])
        
                hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
                hwp.XHwpWindows.Item(0).Visible = True
                # hwp.Open(path + '/[별지  제6호 서식]안전진단 요청을 위한 동의서.hwp')
                hwp.Open(self.form_name[0])
        
                hwp.HAction.Run("MoveTopLevelBegin")
                hwp.HAction.Run("SelectAll")
                hwp.HAction.Run("Copy")
                hwp.HAction.Run("MoveLineBegin")
                hwp.HAction.Run("MoveRight")
                hwp.HAction.Run("TableRightCell")
                hwp.HAction.Run("TableRightCell")
                hwp.HAction.Run("TableRightCell")
                hwp.HAction.Run("TableRightCell")
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HParameterSet.HInsertText.Text = owner[0].split()[0]
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HParameterSet.HInsertText.Text = owner[0].split()[1]
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HParameterSet.HInsertText.Text = address[0]
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
            
                for i in range(len(deungi) - 1):
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("MoveLineEnd")
                    hwp.HAction.Run("MoveRight")
                    hwp.HAction.Run("BreakPage")
                    hwp.HAction.GetDefault("Paste", hwp.HParameterSet.HSelectionOpt.HSet)
                    hwp.HAction.Execute("Paste", hwp.HParameterSet.HSelectionOpt.HSet)
                    hwp.HAction.Run("MoveLineBegin")
                    hwp.HAction.Run("MoveRight")
                    hwp.HAction.Run("TableRightCell")
                    hwp.HAction.Run("TableRightCell")
                    hwp.HAction.Run("TableRightCell")
                    hwp.HAction.Run("TableRightCell")
                    hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HParameterSet.HInsertText.Text = owner[i+1].split()[0]
                    hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HParameterSet.HInsertText.Text = owner[i+1].split()[1]
                    hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HParameterSet.HInsertText.Text = address[i+1]
                    hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    
                # hwp.Quit()
                self.form_name = ''
        
    # 한글양식으로 변환 버튼 클릭
    def convert_to_hangul_form_3(self):
        if self.checkDataFile():
            self.form_name = QFileDialog.getOpenFileName(self)
            
            if self.checkFormFile():
                path = os.getcwd()
                deungi = self.showFile_1(self.fname[0])
                owner = list(deungi['소유자'])
                address = list(deungi['주소'])
                
                hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
                hwp.XHwpWindows.Item(0).Visible = True
                hwp.Open(self.form_name[0])
                # hwp.Open(path + '/[별지 제4호서식] 정비사업 조합설립추진위원회 구성동의서.hwp')
                
                hwp.HAction.Run("MoveTopLevelBegin")
                hwp.HAction.Run("SelectAll")
                hwp.HAction.Run("Copy")
                hwp.HAction.Run("MoveLineBegin")
                hwp.HAction.Run("MoveRight")
                hwp.HAction.Run("TableLowerCell")
                hwp.HAction.Run("TableLowerCell")
                hwp.HAction.Run("TableLowerCell")
                hwp.HAction.Run("TableLowerCell")
                hwp.HAction.Run("TableLowerCell")
                hwp.HAction.Run("TableRightCell")
                hwp.HAction.Run("TableRightCell")
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HParameterSet.HInsertText.Text = owner[0].split()[0]
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HParameterSet.HInsertText.Text = owner[0].split()[1]
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HParameterSet.HInsertText.Text = address[0]
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
            
                for i in range(len(deungi) - 1):
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("MoveLineBegin")
                    hwp.HAction.Run("MoveLeft")
                    hwp.HAction.Run("SelectAll")
                    hwp.HAction.Run("MoveLineEnd")
                    hwp.HAction.Run("BreakPage")
                    hwp.HAction.GetDefault("Paste", hwp.HParameterSet.HSelectionOpt.HSet)
                    hwp.HAction.Execute("Paste", hwp.HParameterSet.HSelectionOpt.HSet)
                    hwp.HAction.Run("MoveSelViewUp")
                    hwp.HAction.Run("MoveLineBegin")
                    hwp.HAction.Run("MoveRight")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableRightCell")
                    hwp.HAction.Run("TableRightCell")
                    hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HParameterSet.HInsertText.Text = owner[i+1].split()[0]
                    hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HParameterSet.HInsertText.Text = owner[i+1].split()[1]
                    hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HParameterSet.HInsertText.Text = address[i+1]
                    hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                        
                # hwp.Quit()
                self.form_name = ''
            
    # 한글양식으로 변환 버튼 클릭
    def convert_to_hangul_form_4(self):
        if self.checkDataFile():
            self.form_name = QFileDialog.getOpenFileName(self)
            
            if checkFormFile():
                path = os.getcwd()
                deungi = self.showFile_1(self.fname[0])
                owner = list(deungi['소유자'])
                address = list(deungi['주소'])
            
                hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
                hwp.XHwpWindows.Item(0).Visible = True
                hwp.Open(self.form_name[0])
                # hwp.Open(path + '/[별지 제6호서식] 조합설립 동의서(재개발사업¸ 재건축사업).hwp')
            
                hwp.HAction.Run("MoveTopLevelBegin")
                hwp.HAction.Run("SelectAll")
                hwp.HAction.Run("Copy")
                hwp.HAction.Run("MoveLineBegin")
                hwp.HAction.Run("MoveRight")
                hwp.HAction.Run("TableLowerCell")
                hwp.HAction.Run("TableLowerCell")
                hwp.HAction.Run("TableLowerCell")
                hwp.HAction.Run("TableLowerCell")
                hwp.HAction.Run("TableLowerCell")
                hwp.HAction.Run("TableRightCell")
                hwp.HAction.Run("TableRightCell")
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HParameterSet.HInsertText.Text = owner[0].split()[0]
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.Run("TableRightCellAppend")
                hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                hwp.HParameterSet.HInsertText.Text = owner[0].split()[1]
                hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
            
                for i in range(len(deungi) - 1):
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableUpperCell")
                    hwp.HAction.Run("TableLeftCell")
                    hwp.HAction.Run("MoveLineBegin")
                    hwp.HAction.Run("MoveLeft")
                    hwp.HAction.Run("SelectAll")
                    hwp.HAction.Run("MoveLineEnd")
                    hwp.HAction.Run("BreakPage")
                    hwp.HAction.GetDefault("Paste", hwp.HParameterSet.HSelectionOpt.HSet)
                    hwp.HAction.Execute("Paste", hwp.HParameterSet.HSelectionOpt.HSet)
                    hwp.HAction.Run("MovePrevParaBegin")
                    hwp.HAction.Run("MovePrevParaBegin")
                    hwp.HAction.Run("MovePrevParaBegin")
                    hwp.HAction.Run("MoveRight")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableLowerCell")
                    hwp.HAction.Run("TableRightCell")
                    hwp.HAction.Run("TableRightCell")
                    hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HParameterSet.HInsertText.Text = owner[i+1].split()[0]
                    hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.Run("TableRightCellAppend")
                    hwp.HAction.GetDefault("InsertText", hwp.HParameterSet.HInsertText.HSet)
                    hwp.HParameterSet.HInsertText.Text = owner[i+1].split()[1]
                    hwp.HAction.Execute("InsertText", hwp.HParameterSet.HInsertText.HSet)
                
                    # hwp.Quit()
                self.form_name = ''
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MyApp()
    gui.show()
    sys.exit(app.exec_())