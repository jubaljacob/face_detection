import cv2

frontal_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#choosen image to detect face test and also conv into grayscale using 0
#img = cv2.imread('grp2.png')

#webcam footage to be captured
webcam = cv2.VideoCapture('vid.mp4')


#itration for frame countinuity 
while True:
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detection
    face_coordinates = frontal_face_data.detectMultiScale(grayscaled_img)

    #drew rectangle (first 4 cords of img)(secind 3 are color of rect) last digit is thickness of rectangle line
    #((x,y) (x+w,y+h)) [x,y,width,height]

    for (x,y,w,h) in face_coordinates:
     cv2.rectangle(frame,(x,y),(x+w, y+h),(225,0,0),2)


    cv2.imshow('Face detection ', frame )
    key = cv2.waitKey(1)
    #STOPS WHEN q is pressed
    if key==81 or key==113:
        break
#release vid capture
webcam.release()

print("Code complete")





#print(face_coordinates)




