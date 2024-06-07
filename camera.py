import urllib.request
import cv2
import numpy as np
import time


if __name__ == '__main__':
    camera_type = 'PC' # PC or IP camera

    if camera_type.lower() == 'pc':
        # Use PC camera and display canny frames. With M1 macbook camera index 0 does not work.
        camera = cv2.VideoCapture(1)
        ret = True
        while ret:
            ret, image = camera.read()
            if ret:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                high_thresh = 255
                bw_edges = cv2.Canny(image_gray, high_thresh * 0.3, high_thresh, L2gradient=True)
                cv2.imshow("Image", bw_edges)
                cv2.waitKey(1)
    else:
        # Using IP Camera
        cap = cv2.VideoCapture(0)
        address = "http://admin:1234@192.168.0.4:8081"  # Your address might be different
        # address = "http://192.168.0.4:8888"  # Your address might be different

        cap.open(address)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Read and display video frames until video is completed or
        # user quits by pressing ESC
        cv2.startWindowThread()
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Display the resulting frame
                cv2.imshow('Frame', frame)
                if (cv2.waitKey(1) & 0xFF == 27):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()