import cv2
import os
import numpy as np
import queue
import threading
import time
import board, busio
import adafruit_mlx90640

# bufferless VideoCapture
class VideoCap:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                        self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()
    
    def release(self):
        self.cap.release()

def center_box(box):
    startX, startY, endX, endY = box.astype("int")
    h = endY - startY
    w = endX - startX
    aspect = h/w
    if aspect > 1:
        offset = int(np.round((h / 2) - (w / 2)))
        return startX - offset, startY, endX + offset, endY
    else:
        offset = int(np.round((w / 2) - (h / 2)))
        return startX, startY - offset, endX, endY + offset

def cycle_calc(time_lst):
    for i in range(len(time_lst) - 1):
        time_lst[i] = time_lst[i + 1] - time_lst[i]
    time_lst.pop(-1)
    return time_lst

possible_thermal_camera_errors = (ValueError, RuntimeError, ZeroDivisionError, OSError)

def tc_bad_connection():
    input('Bad connection to the thermal camera. Try wiggling it and then run the following command: "sudo i2detect -y 1" to see if it\'s connected and then press Enter...')

# Set working dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Face detect model
prototxt_path = '../deploy.prototxt'
weights_path = '../res10_300x300_ssd_iter_140000_fp16.caffemodel'
face_detect = cv2.dnn.readNet(prototxt_path, weights_path)


# Setup I2C
i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)

# Begin MLX90640 with I2C comm
while True:
    try:
        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
        break
    # If there is a bad physical connection to the thermal camera
    except possible_thermal_camera_errors:
        tc_bad_connection()

# Image properties
image_size = 224

cap = VideoCap(0)

cycle_time = []

time1 = 0
time2 = 0

while True:

    frame = cap.read()

    h, w = frame.shape[:2]

    # Setup array for sorting all 768 temperatures
    mlx_frame = np.zeros((24*32,))
    
    while True:
        try:
            mlx.getFrame(mlx_frame)
            break
        # If there is a bad physical connection to the thermal camera
        except possible_thermal_camera_errors:
            tc_bad_connection()

    # Reshape array to matrix
    t_frame = np.reshape(mlx_frame, (24,32))
    # Flip thermal frame
    t_frame = np.flip(t_frame, axis=0)
    # Upscale thermal frame to match camera
    t_frame = t_frame.repeat(20, axis=0).repeat(20, axis=1)

    # Create frame for display alongside camera frame
    # Normailize thermal frame
    t_frame_d = (t_frame - np.min(t_frame)) / np.ptp(t_frame)
    # Round all values in thermal frame
    t_frame_d = np.round(t_frame_d * 255, 0).astype(np.uint8)
    # # Add dimensions to frame to make it appear as RGB
    t_frame_d = np.stack((t_frame_d,)*3, axis=-1)

    # Flip camera frame
    frame = cv2.flip(frame, -1)
    # Crop and resize camera frame to try and match thermal frame
    frame = cv2.resize(frame[20:400, 40:527], (640,480))

    # OpenCV DNN pre-processing
    blob = cv2.dnn.blobFromImage(frame, 1.0, (image_size, image_size))
    # Process the pre-processed frame & find faces
    face_detect.setInput(blob)
    faces = face_detect.forward()

    # For all faces detected in frame
    for i in range(0, faces.shape[2]):
        # Get the confidence that it is a face 
        confidence = faces[0, 0, i, 2]
        if confidence > 0.7:

            # Get coordinates for face
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = center_box(box)

            # ensure the bounding boxes fall within the dimensions of the frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            # Pre-process frame
            frame_crop = frame[startY:endY, startX:endX]
            frame_crop_resize = cv2.resize(frame_crop, (image_size, image_size), cv2.INTER_AREA)
            frame_crop_resize_reshape = np.array(frame_crop_resize, dtype=np.float32).reshape(-1, image_size, image_size, 3)

            # Crop thermal frame of face
            t_frame_d_crop = t_frame[startY:endY, startX:endX]
            # Calculate mean temperature of cropped thermal frame
            temp = np.round(np.max(t_frame_d_crop), 1)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
            cv2.rectangle(t_frame_d, (startX, startY), (endX, endY), (255,0,255), 2)
            cv2.putText(t_frame_d, str(temp), (startX, startY), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)

    frame = np.concatenate((frame, t_frame_d), axis=0)

    time1 = time.time()
    cycle_time.append(time1)
    fps = 1 / (time1 - time2)
    time2 = time1

    cv2.putText(frame, str(int(fps)) + 'fps' + str(np.max(t_frame)), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow('window', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

mean_cycles = np.round(np.mean(cycle_calc(cycle_time)), 6) * 1000

print(mean_cycles)

cap.release()
cv2.destroyAllWindows()
