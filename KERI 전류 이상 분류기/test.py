import cv2
import numpy as np

def setLabel(image, str, contour):
        (text_width, text_height), baseline = cv2.getTextSize(str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        x, y, width, height = cv2.boundingRect(contour)
        pt_x = x + int((width-text_width)/2)
        pt_y = y + int((height+text_height)/2)
        cv2.rectangle(image, (pt_x, pt_y+baseline), (pt_x + text_width, pt_y-text_height), (200, 200, 200), cv2.FILLED)
        cv2.putText(image, str, (pt_x, pt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, 8)

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
                cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 3)
                size = len(cnt)
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                size = len(approx)

                if size == 3:
                        setLabel(frame, "triangle", cnt)
                elif size == 4:
                        setLabel(frame, "rectangle", cnt)
                elif size == 5:
                    setLabel(frame, "pentagon", cnt)
                elif size == 6:
                    setLabel(frame, "hexagon", cnt)
                else:
                    setLabel(frame, str(size), cnt)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


