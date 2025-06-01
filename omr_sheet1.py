import cv2
import numpy as np
import utiliss

#######################
path = "omr_sheetF2.jpg"
heightImg=600
widthImg=600
questions=10
choices=5
ans = [4,4,4,4,4,4,4,4,4,4]
webcamfeed = True
#######################
cap = cv2.VideoCapture(0)
cap.set(10,130)

while True:
    if webcamfeed:
        success, img = cap.read()
    else:
        img = cv2.imread(path)


    #Preprocessing
    img = cv2.resize(img, (widthImg,heightImg))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBiggestcontours = img.copy()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,10,50)

    try:
        #Finding  countours
        contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours,contours,-1,(0,255,0),10)
        #Finding Rectangles
        rectcon = utiliss.rectcontours(contours)
        biggestcontour = utiliss.getcornerpoints(rectcon[0])
        gradepoints = utiliss.getcornerpoints(rectcon[2])
        #print(biggestcontour)

        if biggestcontour.size !=0 and gradepoints.size !=0:
            cv2.drawContours(imgBiggestcontours,biggestcontour,-1,(0,255,0),15)
            cv2.drawContours(imgBiggestcontours,gradepoints,-1,(255,0,0),15)

            biggestcontour= utiliss.reorder(biggestcontour)
            gradepoints = utiliss.reorder(gradepoints)

            pt1 = np.float32(biggestcontour)
            pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1,pt2)
            imgWarpcolored= cv2.warpPerspective(img,matrix,(widthImg,heightImg))

            ptG1 = np.float32(gradepoints)
            ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
            #cv2.imshow("Grade",imgGradeDisplay)

            #Applying Threshold
            imgWarpGray= cv2.cvtColor(imgWarpcolored,cv2.COLOR_BGR2GRAY)
            imgThresh= cv2.threshold(imgWarpGray,100,255,cv2.THRESH_BINARY_INV)[1]

            boxes= utiliss.splitboxes(imgThresh)
            #cv2.imshow("test",boxes[37])
            #print(cv2.countNonZero(boxes[7]),cv2.countNonZero(boxes[0]))

            #Obtaining non zero pixel values of each box
            myPixelVal= np.zeros((questions,choices))
            countC= 0
            countR= 0

            for image in boxes:
                totalpixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalpixels
                countC +=1
                if (countC == choices):countR +=1 ;countC=0
                #print(myPixelVal)

            #Finding Index Values of the markings
            myIndex = []
            for x in range (0,questions):
                arr = myPixelVal[x]
                #print("arr",arr)
                myIndexVal = np.where(arr==np.amax(arr))
                #print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])
            #print(myIndex)

            #GRADING
            grading=[]
            for x in range(0,questions):
                if ans [x]== myIndex[x]:
                    grading.append(1)
                else:grading.append(0)
            #print(grading)
            score= (sum(grading)/questions) *100 #FINAL GRADE
            print(score)

            #Displaying Answers
            imgResult = imgWarpcolored.copy()
            imgResult = utiliss.showanswers(imgResult,myIndex,grading,ans,questions,choices)
            imgRawDrawing = np.zeros_like(imgWarpcolored)
            imgRawDrawing= utiliss.showanswers(imgRawDrawing,myIndex,grading,ans,questions,choices)
            invmatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imginvwarp = cv2.warpPerspective(imgRawDrawing, invmatrix, (widthImg, heightImg))

            imgRawGrade = np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade,str(int(score))+"%",(50,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),3)
            InvmatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            imgInvGrade = cv2.warpPerspective(imgRawGrade, InvmatrixG, (widthImg, heightImg))

            imgFinal = cv2.addWeighted(imgFinal,1,imginvwarp,1,0)
            imgFinal = cv2.addWeighted(imgFinal,1,imgInvGrade,1,0)



        imgBlank = np.zeros_like(img)
        imageArray = ([img,imgGray,imgBlur,imgCanny],
                      [imgContours,imgBiggestcontours,imgWarpcolored,imgThresh],
                      [imgFinal,imgRawDrawing,imgResult,imgBlank])
    except:
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgBlank, imgBlank, imgBlank, imgBlank],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    lables = [["Original","Gray","Blur","Canny"],
              ["Contours","Biggest Con","Warp","Threshold"],
              ["Final Image","","",""]]
    imgStacked = utiliss.stackImages (imageArray,0.35)


    cv2.imshow("Final Result",imgFinal)
    cv2.imshow("Stacked Images",imgStacked)
    if cv2.waitKey(1) & 0xFF == ord('u'):
        cv2.imwrite("FinalResult.jpg",imgFinal)
        cv2.waitKey(300)
        cv2.destroyAllWindows()