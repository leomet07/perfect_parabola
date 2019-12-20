import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
	

	denom = (x1-x2) * (x1-x3) * (x2-x3)
	A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
	B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
	C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

	return A,B,C

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
total_detected = []
img_shape = []
vid = 0
cap = cv2.VideoCapture(vid)
photos_taken = 0
last_frame = None
while cap.isOpened():
    ret, frame = cap.read()
    img_shape = frame.shape

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
    elif key == ord(" "):
        last_frame = frame.copy()
        #add detection here
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            total_detected.append( (int(x), int(y)))
            if True:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        print(photos_taken)
        cv2.imshow("Frame",frame)
        cv2.waitKey(0)

        #if ball detected
        if len(cnts) > 0:
            photos_taken += 1

    if photos_taken == 3:
        
        break


cv2.destroyAllWindows()





print(img_shape)
print(total_detected)
screen_w = img_shape[1]
screen_h = img_shape[0]
(x1,y1),(x2,y2),(x3,y3) = total_detected

#shopw a frame

cv2.circle(last_frame, (int(x1) , int(y1)), 5, (255, 0, 0), -1)

cv2.circle(last_frame, (int(x2) , int(y2)), 5, (0, 255, 0), -1)

cv2.circle(last_frame, (int(x3) , int(y3)), 5, (0, 0, 255), -1)




#invert the y bc cv2 goes from top left and not bottom right liek a nroaml pl,ot
y1 = screen_h - y1
y2 = screen_h - y2
y3 = screen_h - y3
total = [x1,x2,x3]
print(total)
#Calculate the unknowns of the equation y=ax^2+bx+c
a,b,c=calc_parabola_vertex(x1, y1, x2, y2, x3, y3)


#Define x range for which to calc parabola


x_pos=np.arange(0,screen_w,1)
y_pos=[]

#Calculate y values 
for x in range(len(x_pos)):
	x_val=x_pos[x]
	y=(a*(x_val**2))+(b*x_val)+c
	y_pos.append(y)


# Plot the parabola (+ the known points)


plt.plot(x_pos, y_pos) # parabola line
#to draw each point
#plt.scatter(x_pos, y_pos, color='gray') # parabola points
plt.scatter(x1,y1,color='r',marker="D",s=50) # 1st known xy
plt.scatter(x2,y2,color='g',marker="D",s=50) # 2nd known xy
plt.scatter(x3,y3,color='b',marker="D",s=50) # 3rd known xy
plt.show()

#manually draw the parabola
for i in range(len(x_pos)):
    x_pix = int(round(x_pos[i]))
    y_pix = int(round(y_pos[i]))


    #invert back to normal values(back to starting to top left, and not bottom left like a graph)
    y_pix = screen_h - y_pix
    #print(x_pix,y_pix)

    red = [0,0,255]

    # Change one pixel
    try:
        last_frame[y_pix,x_pix]=red
    except:
        pass
    

cv2.imshow("Manual line",last_frame)
cv2.waitKey(0)


cv2.imwrite("predicted.png",last_frame)