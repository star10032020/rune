import math
import time
from ultralytics import YOLO
import numpy as np
import cv2


class rune_detect:
    model = YOLO("")
    targetColor = 0

    def __init__(self,model_path,targetColor,imgsz):#颜色0为识别蓝色 颜色1为识别红色
        self.BoolGpu = False

        import torch
        torch_cuda=torch.cuda.is_available()
        cv2_count=cv2.cuda.getCudaEnabledDeviceCount()
        if torch_cuda == True:
            if cv2_count > 0:
                self.BoolGpu = True
                cv2.cuda.setDevice(0)
                
        self.YoloTime=0.00
        self.afterTime=0.00
        self.model = YOLO(model_path)
        self.targetColor = targetColor
        self.detectSize=imgsz
        self.thresh3 = 76
        print("model have loaded")
    def detect(self,image):#返回的是外部俩点与内部俩点，相对于R都是顺时针方向
        return self.splitOne(image)

    def red_or_blue(self,image,x,y,w,h):#统计图片是蓝色多还是红色多。红色返回1，蓝色返回0
        
        if self.BoolGpu == False:
            roi = image[y:y + h, x:x + w]

            # 定义红色和蓝色的颜色范围
            lower_red = np.array([0, 0, 200])
            upper_red = np.array([50, 50, 255])

            lower_blue = np.array([200, 0, 0])
            upper_blue = np.array([255, 50, 50])

            # 创建掩模
            mask_red = cv2.inRange(roi, lower_red, upper_red)
            mask_blue = cv2.inRange(roi, lower_blue, upper_blue)

            # 计算红色和蓝色像素数量
            red_count = np.sum(mask_red == 255)
            blue_count = np.sum(mask_blue == 255)
        else:
            roi = image[y:y + h, x:x + w]
           # 定义红色和蓝色的颜色范围
            lower_red = np.array([0, 0, 200])
            upper_red = np.array([50, 50, 255])

            lower_blue = np.array([200, 0, 0])
            upper_blue = np.array([255, 50, 50])

            # 创建掩模
            mask_red = cv2.inRange(roi, lower_red, upper_red)
            mask_blue = cv2.inRange(roi, lower_blue, upper_blue)

            # 计算红色和蓝色像素数量
            red_count = np.sum(mask_red == 255)
            blue_count = np.sum(mask_blue == 255)

        if red_count>blue_count:
            return 1 #红色为主
        else:
            return 0 #蓝色为主
        return 3 #未定义结果
    def get_same_area(self, rect1, rect2):
        points1 = cv2.boxPoints(rect1)
        points2 = cv2.boxPoints(rect2)

        poly1 = np.array(points1, dtype=np.int32)
        poly2 = np.array(points2, dtype=np.int32)
        intersection = cv2.intersectConvexConvex(poly1,poly2)
        return intersection[0]
    def get_same_point(self,rect1,rect2, RItem):#先是内侧的R向外看的右，左;再是外侧的右，左
        ansList=[]
        same_point1 = (0, 0)
        same_point2 = (0, 0)

        vertices1f = cv2.boxPoints(rect1)
        vertices2f = cv2.boxPoints(rect2)

        vertices1 = np.array(vertices1f,dtype=np.int32)
        vertices2 = np.array(vertices2f, dtype=np.int32)

        center1f = rect1[0]
        center2f = rect2[0]
        center1 = (int(center1f[0]),int(center1f[1]))
        center2 = (int(center2f[0]), int(center2f[1]))

        inner1 = sorted(vertices1, key=lambda point: np.linalg.norm(point - center2))#得到靠近内心的4个点
        inner2 = sorted(vertices2, key=lambda point: np.linalg.norm(point - center1))

        focusX=0.0
        focusY=0.0

        focusX = inner1[0][0]+inner1[1][0]+inner2[0][0]+inner2[1][0]
        focusX = focusX/4.0

        focusY = inner1[0][1] + inner1[1][1]+inner2[0][1]+inner2[1][1]
        focusY = focusY/4.0
        clsR, xywhR = RItem
        Rx = xywhR[0]+xywhR[2]/2.0
        Ry = xywhR[1]+xywhR[3]/2.0
        micVec =(int(focusX-Rx),int(focusY-Ry))

        targetVec = (int(inner1[0][0]-Rx),int(inner1[0][1]-Ry))
        if micVec[0]*targetVec[1]-micVec[1]*targetVec[0]>=0:
            ansList.append(inner1[0])
            ansList.append(inner1[1])
        else:
            ansList.append(inner1[1])
            ansList.append(inner1[0])

        targetVec = (int(inner2[0][0] - Rx), int(inner2[0][1] - Ry))
        if micVec[0] * targetVec[1] - micVec[1] * targetVec[0] >= 0:
            ansList.append(inner2[0])
            ansList.append(inner2[1])
        else:
            ansList.append(inner2[1])
            ansList.append(inner2[0])
        #ansList.append((int(Rx),int(Ry)))#R
        return ansList







    def getNeedPoint(self,image, RItem, BigItem, LittleItem):
        #cv2.imshow("littleImage",image)
        #cv2.waitKey(1000)
        ansList = []
        video1 = image.copy()
        alpha = 0.6
        beta = -20
        video1 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        gray = cv2.cvtColor(video1, cv2.COLOR_BGR2GRAY)
        gray0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_thresh = cv2.threshold(gray, self.thresh3, 255, cv2.THRESH_BINARY)[1]
        gray_thresh2 = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)[1]
        canny0 = cv2.Canny(gray, 0, 240)
        gray_bit = gray_thresh.copy()

        contours, heri = cv2.findContours(gray_bit,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        gray_bit2=gray_bit.copy()
        #imag=cv2.drawContours(gray_bit2,contours,-1,(255,255,255),1)
        #cv2.imshow("imag",imag)
        #cv2.waitKey(1000)
        max_perimeter1, max_perimeter2 = -1, -1
        max_id1,max_id2 = -1, -1
        #print(len(contours))
        for i in range(len(contours)):
            countor = contours[i]
            contour_area = cv2.contourArea(countor)
            contour_perimeter = cv2.arcLength(countor,True)
            rect = cv2.minAreaRect(countor)
            #print(rect)
            if int(rect[1][0]*rect[1][1])==0:continue
            rate = contour_area*1.0/(rect[1][0]*rect[1][1])
            if abs(rect[1][0]-rect[1][1])<6:
                continue
            if rate > 0.7 or rate < 0.1:
                continue
            if contour_perimeter > max_perimeter1:
                max_id1 = i
                max_perimeter1 = contour_perimeter
        if max_id1 == -1:
            return ansList
        rect1 = cv2.minAreaRect(contours[max_id1])#判断为马蹄铁靠近R那部分
        for i in range(len(contours)):
            countor = contours[i]
            contour_area = cv2.contourArea(countor)
            contour_perimeter = cv2.arcLength(countor,True)
            rect = cv2.minAreaRect(countor)
            if int(rect[1][0]*rect[1][1])==0:continue
            rate = contour_area*1.0/(rect[1][0]*rect[1][1])
            if abs(rect[1][0]-rect[1][1]) < 6:
                continue
            center1, center2 =(int(rect1[0][0]),int(rect1[0][1])),(int(rect[0][0]),int(rect[0][1]))
            littledis = math.sqrt((center1[0]-center2[0])*(center1[0]-center2[0])+(center1[1]-center2[1])*(center1[1]-center2[1]))
            if littledis < max(min(center1[0],center1[1]),min(center2[0],center2[1]))*0.8:
                #print("littledis is too litle,so we continue")
                continue #在python里柑橘不好用
            if self.get_same_area(rect1,rect)>=min(rect1[1][1]*rect1[1][0],rect[1][1]*rect[1][0])*0.5:
                #print("same_area is too big,so we continue")
                #print("same_area is "+str(self.get_same_area(rect1,rect)))
                #print("rect1_area is " + str(rect1[1][1]*rect1[1][0]))
                #print("rect_area is " + str(rect[1][1]*rect[1][0]))
                continue
            if rate > 0.7 or rate < 0.1:
                #print("rate is too big or too few,so we continue")
                continue
            #print("contour_perimeter="+str(contour_perimeter)+",max_perimeter2="+str(max_perimeter2)+",max_perimeter1="+str(max_perimeter1))
            if contour_perimeter > max_perimeter2 and contour_perimeter<max_perimeter1:
                max_id2 = i
                max_perimeter2 = contour_perimeter
        #print("max_id1="+str(max_id1))
        #print("max_id2=" + str(max_id2))
        if max_id1 == -1 or max_id2 == -1:
            return ansList
        # 找到了马蹄铁靠外那部分
        rect2 = cv2.minAreaRect(contours[max_id2])

        point1 =cv2.boxPoints(rect1).astype(int)
        point2 = cv2.boxPoints(rect2).astype(int)
        #print("we have try get_same_point")
        tempans = self.get_same_point(rect1, rect2,RItem)
        for item in tempans:
            ansList.append(item)

        return ansList





    def overlap_area(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # 计算水平和垂直方向上的重叠部分
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

        # 计算重叠面积
        area = overlap_x * overlap_y

        return area

    def splitOne(self,image):
        ansList=[]

        start_time = time.time()
        
        results = self.model(source=image,imgsz=self.detectSize)
        
        end_time = time.time()
        self.YoloTime+=end_time-start_time
        
        start_time = time.time()

        RList = []
        BigList = []
        LittleList = []
        for result in results:
            #im0=result.plot()
            #cv2.imshow("plot",im0)
            #cv2.waitKey(1000)
            boxList = result.boxes.xywh.tolist()
            clsList = result.boxes.cls.tolist()

            size = len(boxList)
            for i in range(size):
                xywh0 = boxList[i]
                xywh=(xywh0[0]-xywh0[2]/2.0,xywh0[1]-xywh0[3]/2.0,xywh0[2],xywh0[3])#回到opencv定义
                cls = int(clsList[i])
                if self.targetColor != self.red_or_blue(image,int(xywh[0]),int(xywh[1]),int(xywh[2]),int(xywh[3])):
                    continue

                if cls == 0:#R
                    RList.append((cls, xywh))
                if cls == 1:#BigRoi
                    BigList.append((cls, xywh))
                if cls == 2:#LittleRoi
                    LittleList.append((cls, xywh))
            #得到了颜色相同的识别结果
            RightRList = []
            RightBigList = []
            RightLittleList = []
            for littleItem in LittleList:
                clslittle, xywhlittle = littleItem
                for BigItem in BigList:
                    clsBig, xywhBig = BigItem
                    same_area = self.overlap_area(xywhlittle, xywhBig)
                    if int(xywhlittle[2]*xywhlittle[3])==0:continue
                    same_rate = same_area*1.0/(xywhlittle[2]*xywhlittle[3])
                    if same_rate > 0.8:#视为正确的LittleRoi
                        RightBigList.append(BigItem)
                        RightLittleList.append(littleItem)
                        break
            #print("length of RightBigList:"+str(len(RightBigList)))
            #print("length of RightBigList:" + str(len(RightLittleList)))
            #print("length of RList:" + str(len(RList)))
            for RItem in RList:
                clsR,xywhR = RItem
                midxR = xywhR[0]+xywhR[2]/2.0
                midyR = xywhR[1]+xywhR[3]/2.0
                for i in range(len(RightBigList)):
                    BigItem = RightBigList[i]
                    littleItem = RightLittleList[i]
                    clsBig,xywhBig = BigItem
                    midxBig = xywhBig[0]+xywhBig[2]/2.0
                    midyBig = xywhBig[1]+xywhBig[3]/2.0
                    dis = math.sqrt((midxR-midxBig)*(midxR-midxBig)+(midyR-midyBig)*(midyR-midyBig))
                    if dis < min(xywhBig[2],xywhBig[3])*1.2:#应该足够近
                        clslittle,xywhlittle = littleItem

                        image2 = image[int(xywhlittle[1]):int(xywhlittle[1]) + int(xywhlittle[3]),
                                 int(xywhlittle[0]):int(xywhlittle[0]) + int(xywhlittle[1])]
                        #print("we have trying getNeedPoint")
                        #注意，由于坐标系变换，RItem应该减一下
                        xywhR2=(xywhR[0]-int(xywhlittle[0]),xywhR[1]-int(xywhlittle[1]),xywhR[2],xywhR[3])
                        RItem2=(clsR,xywhR2)

                        tempans = self.getNeedPoint(image2, RItem2, BigItem, littleItem)
                        if len(tempans)<=0:
                            continue
                        #tempans直接返回的坐标系是以小图左上角为（0，0）的，需要转换
                        for tempItem in tempans:
                            ansList.append((tempItem[0]+int(xywhlittle[0]),tempItem[1]+int(xywhlittle[1])))
                        end_time = time.time()
                        self.afterTime+=end_time-start_time
                        return ansList
        end_time = time.time()
        self.afterTime+=end_time-start_time
        return ansList





