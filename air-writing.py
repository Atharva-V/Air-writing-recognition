import cv2
import numpy as np
import copy
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import time
import autopy
import math
import mediapipe as mp

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 26)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Adding sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def get_histogram(frame):
    roi1 = frame[rect1_tl[1]:rect1_tl[1] + width, rect1_tl[0]:rect1_tl[0] + height]
    roi2 = frame[rect2_tl[1]:rect2_tl[1] + width, rect2_tl[0]:rect2_tl[0] + height]
    roi3 = frame[rect3_tl[1]:rect3_tl[1] + width, rect3_tl[0]:rect3_tl[0] + height]
    roi4 = frame[rect4_tl[1]:rect4_tl[1] + width, rect4_tl[0]:rect4_tl[0] + height]
    roi5 = frame[rect5_tl[1]:rect5_tl[1] + width, rect5_tl[0]:rect5_tl[0] + height]
    roi = np.concatenate((roi1, roi2, roi3, roi4, roi5), axis=0)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    return cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

def get_ROI(canvas):
    gray = cv2.bitwise_not(canvas)
    ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for i in range(len(ctrs)):
        x, y, w, h = cv2.boundingRect(ctrs[i])
        areas.append((w * h, i))

    def sort_second(val):
        return val[0]

    areas.sort(key=sort_second, reverse=True)
    x, y, w, h = cv2.boundingRect(ctrs[areas[1][1]])
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 1)
    roi = gray[y:y + h, x:x + w]
    return roi

def character_prediction(roi, model):
    """Predicts character written with image processing"""
    img = cv2.resize(roi, (28, 28))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = Image.fromarray(img)

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))
    ])

    p_img = preprocess(img)

    model.eval()
    p_img = p_img.reshape([1, 1, 28, 28]).float()
    output = model(torch.transpose(p_img, 2, 3))
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds
class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.8, trackCon=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    canvas = np.zeros((720, 1280), np.uint8)
    far_points = []
    pressed = False
    is_drawing = False
    made_prediction = False
    # Creating the model
    model = Cnn()
    model.load_state_dict(torch.load('model_emnist.pt', map_location='cpu'))
    while True:
        _, frame = cap.read()
        original_frame = copy.deepcopy(frame)
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[8][1:])
            #far = lmList[8][1:]
            far = (lmList[8][1],lmList[8][2])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        
        canvas[:, :] = 255

        key = cv2.waitKey(1)

        # ready to draw
        if key & 0xFF == ord('s'):
            pressed = True
            histogram = get_histogram(frame)

        # To start drawing
        if key & 0xFF == ord('d'):
            is_drawing = True

        # To clear drawing
        if key & 0xFF == ord('c'):
            canvas[:, :] = 255
            is_drawing = False
            far_points.clear()
            made_prediction = False

        if is_drawing:
            if len(far_points) > 100:
                far_points.pop(0)
            far_points.append(far)
            print(far)
            for i in range(len(far_points) - 1):
                cv2.line(original_frame, far_points[i], far_points[i + 1], (255, 5, 255), 20)
                cv2.line(canvas, far_points[i], far_points[i + 1], (0, 0, 0), 20)

        # To predict the character drawn
        if key & 0xFF == ord('p'):
            is_drawing = False
            roi = get_ROI(canvas)
            print(roi)
            prediction = character_prediction(roi, model)
            print(prediction)
            made_prediction = True
            name = str(prediction) + '.jpg'
            cv2.imwrite(name, roi)

        if pressed:
            mask = get_mask(frame, histogram)
            max_contour = get_max_contour(mask)
            hull = cv2.convexHull(max_contour, returnPoints=False)
            draw_defects(original_frame, max_contour, hull)
            defects = cv2.convexityDefects(max_contour, hull)
            far = get_farthest_point(defects, max_contour, get_centroid(max_contour))
            cv2.circle(original_frame, far, 10, [0, 200, 255], -1)

        if made_prediction:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original_frame,"Character Written: " + chr(prediction + 65), (8, 250), font, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            canvas[:, :] = 255
            is_drawing = False
            far_points.clear()
            
        # To quit the drawing
        if key & 0xFF == ord('q'):
            break

        cv2.imshow("frame", original_frame)
        cv2.waitKey(1)
        
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main()