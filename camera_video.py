import cv2 as cv
import time
import geocoder
import os

class_name = []
with open(os.path.join("C:\\Users\purus\OneDrive\Desktop\poth\project_files",'C:\\Users\purus\OneDrive\Desktop\poth\project_files\obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net1 = cv.dnn.readNet('C:\\Users\purus\OneDrive\Desktop\poth\project_files\yolov4_tiny.weights', 'C:\\Users\purus\OneDrive\Desktop\poth\project_files\yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

cap = cv.VideoCapture(0) 
width  = cap.get(3)
height = cap.get(4)
result = cv.VideoWriter('result.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         10,(int(width),int(height)))

g = geocoder.ip('me')
result_path = "C:\\Users\purus\OneDrive\Desktop\poth\pothole_coordinates"
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.1
frame_counter = 0
i = 0
b = 0

while True:
    ret, frame = cap.read()
    frame_counter += 1
    if ret == False:
        break

    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w*h
        area = width*height
        if(len(scores)!=0 and scores[0]>=0.7):
            if((recarea/area)<=0.1 and box[1]<600):
                cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 1)
                cv.putText(frame, "%" + str(round(scores[0]*100,2)) + " " + label, (box[0], box[1]-10),cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
                if(i==0):
                    cv.imwrite(os.path.join(result_path,'pothole'+str(i)+'.jpg'), frame)
                    with open(os.path.join(result_path,'pothole'+str(i)+'.txt'), 'w') as f:
                        f.write(str(g.latlng))
                        i=i+1
                if(i!=0):
                    if((time.time()-b)>=2):
                        cv.imwrite(os.path.join(result_path,'pothole'+str(i)+'.jpg'), frame)
                        with open(os.path.join(result_path,'pothole'+str(i)+'.txt'), 'w') as f:
                            f.write(str(g.latlng))
                            b = time.time()
                            i = i+1
            
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    cv.putText(frame, f'FPS: {fps}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    cv.imshow('frame', frame)
    result.write(frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
print("The width is {}".format(w))
width=int(input("Enter car width"))

if w >= width :
    lane=int(input(""))
    if (lane==0):
        from tkinter import *
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image,ImageTk
        def close_window():
            root.destroy()
        
        root=Tk()
        label=ttk.Label(root,text="Turn left")
        label.config(font=('Arial',20,'bold'))
        label.pack()
        img=Image.open("C:\\Users\purus\OneDrive\Desktop\left.jpg")
        photo=ImageTk.PhotoImage(img)
        label=tk.Label(root,image=photo)
        label.pack()
        root.after(6000,close_window)
        root.mainloop()
    else :
        from tkinter import *
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image,ImageTk
        def close_window():
            root.destroy()
        root=Tk()
        label=ttk.Label(root,text="Turn Right")
        label.config(font=('Arial',20,'bold'))
        label.pack()
        img=Image.open("C:\\Users\purus\OneDrive\Desktop\Right.jpg")
        photo=ImageTk.PhotoImage(img)
        label=tk.Label(root,image=photo)
        label.pack()
        
        root.after(6000,close_windoW)
        root.mainloop()
    

else:
    from tkinter import *
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image,ImageTk
    root=Tk()
    label=ttk.Label(root,text="Go Straight")
    label.config(font=('Arial',20,'bold'))
    label.pack()
    img=Image.open("C:\\Users\purus\OneDrive\Desktop\sum.jpg")
    photo=ImageTk.PhotoImage(img)
    label=tk.Label(root,image=photo)
    label.pack()
    root.mainloop()
        
cap.release()
result.release()
cv.destroyAllWindows()
import cv2
import numpy as np

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny

def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)

    triangle = np.array([[
    (200, height),
    (550, 250),
    (1100, height),]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# lane_canny = canny(lane_image)
# cropped_canny = region_of_interest(lane_canny)
# lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
# averaged_lines = average_slope_intercept(image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 0)

#
cap = cv2.VideoCapture("test.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

