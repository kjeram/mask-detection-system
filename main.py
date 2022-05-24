from json import load
import cv2
import os
import numpy as np
import queue
import threading
import time

# ====== CONFIG BEGINS ====== #  

# Set working dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set to True if running on Raspberry Pi
arm = True

# Face detect model
prototxt_path = 'deploy.prototxt'
weights_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
face_detect = cv2.dnn.readNet(prototxt_path, weights_path)

# The strictness/laxness of the  
# face detction considers a face
face_detect_threshold = 0.7

# CNN
model_name = 'model'
model_2_name = 'model_2'
unique_labels = ['no mask', 'incorrect mask', 'correct mask']

# Image properties
image_size = 224

# Amount of cycles the system will do before making a prediction
# NOTE: should be an odd number
cycles = 799

# For label color in display
# Red, Yellow, Green
color = [(0,0,255), (0,255,255), (0,255,0)]

# ====== CONFIG ENDS ====== #  

# Bufferless VideoCapture
# Should solve the issue of lagging frames
class VideoCap:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, 
    # keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:    # discard previous (unprocessed) frame
                        self.q.get_nowait() 
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

def classify_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = image
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def cycle_calc(time_lst):
    for i in range(len(time_lst) - 1):
        time_lst[i] = time_lst[i + 1] - time_lst[i]
    time_lst.pop(-1)
    return time_lst

possible_thermal_camera_errors = (OSError, RuntimeError, ValueError, ZeroDivisionError)

def tc_bad_connection():
    input('Bad connection to the thermal camera. Try wiggling it and then run the following command: "sudo i2detect -y 1" to see if it\'s connected and then press Enter...')

# CNN
if arm:
    from tflite_runtime.interpreter import Interpreter
    model = Interpreter(model_name + '.tflite')
    model.allocate_tensors()
    model_2 = Interpreter(model_2_name + '.tflite')
    model_2.allocate_tensors()
else:
    from tensorflow.keras.models import load_model
    model = load_model(model_name)
    model_2 = load_model(model_2_name)

# Thermal camera
if arm:
    import board, busio
    import adafruit_mlx90640

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

# Used for calculating fps and cycle time
time1 = 0
time2 = 0

# Holds prediction results
predictions = []
if arm:
    body_temperature = []
else:
    mean_body_temp = 0
cycle_time = []

# Camera init
video_capture = VideoCap(0)

while len(predictions) <= cycles:

    results_mask = np.array([]).astype(np.int32)

    # Read frame from camera
    frame = video_capture.read()
    h, w = frame.shape[:2]

    if arm:
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
        t_frame_org = np.reshape(mlx_frame, (24,32))

        # Flip thermal frame
        t_frame_org = np.flip(t_frame_org, axis=0)

        # Upscale thermal frame to match camera
        t_frame_org = t_frame_org.repeat(20, axis=0).repeat(20, axis=1)

        # Create frame for display alongside camera frame
        # Normailize thermal frame
        t_frame = (t_frame_org - np.min(t_frame_org)) / np.ptp(t_frame_org)

        # Round all values in thermal frame
        t_frame = np.round(t_frame * 255, 0).astype(np.uint8)

        # # Add dimensions to frame to make it appear as RGB
        t_frame = np.stack((t_frame,)*3, axis=-1)

        # Flip camera frame
        frame = cv2.flip(frame, -1)
        # Crop and resize camera frame to match the thermal frame
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

        if confidence > face_detect_threshold:

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

            # Used for displaying the right mask and color information
            index_offset = 0

            # Make a prediction
            if arm:
                pred = classify_image(model, frame_crop_resize_reshape)
                pred_index = np.argmax(pred[0])
                if pred_index == 1:
                    pred = classify_image(model_2, frame_crop_resize_reshape)
                    # Convert prediction to lables
                    pred_index = np.argmax(pred[0])
                    index_offset = 1
            else:
                pred = model.predict(frame_crop_resize_reshape)
                pred_index = np.argmax(pred[0])
                if pred_index == 1:
                    pred = model_2.predict(frame_crop_resize_reshape)
                    # Convert prediction to lables
                    pred_index = np.argmax(pred[0])
                    index_offset = 1

            # Append the prediction to the results
            results_mask = np.append(results_mask, pred_index + index_offset)
            
            # Draw a rectangle around the faces with classification
            text = unique_labels[pred_index + index_offset] + ' (' +  str(round(pred[0][pred_index] * 100)) + '%)'
            cv2.rectangle(frame, (startX, startY), (endX, endY), color[pred_index + index_offset], 2)
            cv2.putText(frame, text, (startX, startY - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, color[pred_index + index_offset], 2, cv2.LINE_AA)

            if arm:
                # Crop thermal frame of face
                t_frame_org_crop = t_frame_org[startY:endY, startX:endX]

                # Get max temp of cropped thermal frame
                temp = np.round(np.max(t_frame_org_crop), 1)

                # Draw text on thermal frame
                cv2.rectangle(t_frame, (startX, startY), (endX, endY), (0,153,255), 2)
                cv2.putText(t_frame, str(temp), (startX, startY), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,153,255), 2, cv2.LINE_AA)
            
            # Record the prediction
            predictions.append(unique_labels[pred_index])
            if arm: body_temperature.append(temp)
    
    # Combine the regular and thermal frame into one
    if arm:
        frame = np.concatenate((frame, t_frame), axis=0)

    # Calculate fps and cycle time
    time1 = time.time()
    cycle_time.append(time1)
    fps = 1 / (time1 - time2)
    time2 = time1

    # Draw fps on frame
    cv2.putText(frame, str(int(fps)) + 'fps', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
        
    # Display the resulting frame
    cv2.imshow('window', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Calculate the final results
max_pred = max(predictions, key=predictions.count)
if arm:
    mean_body_temp = np.round(np.max(body_temperature), 1)
mean_cycles = np.round(np.mean(cycle_calc(cycle_time)), 6) * 1000

# Print results
print('Mask: ' + max_pred)
print('Temp: ' + str(mean_body_temp))
print('Cycle: ' + str(mean_cycles))

# Keep the last display frame
cv2.waitKey(0)

# Release camera and close display window 
video_capture.release()
cv2.destroyAllWindows()