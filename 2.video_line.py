import cv2
cap = cv2.VideoCapture('-331.mp4')
while True:
    ret, frame = cap.read()
    if ret:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height), True)
        cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
        cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
        cv2.line(frame, (0, 1400), (2000,1400), (0,0, 255),5)
        #cv2.rectangle(frame, (100,100), (300,300), (0, 255, 0), 3)
        #cv2.circle(frame, (200,200), 100, (255, 0, 0), 3)
        cv2.imshow('image', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
