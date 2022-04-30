import cv2
import os
import numpy as np
import time

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

# Set to True if running on Raspberry Pi
arm = False

# Set working dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Face detect model
prototxt_path = 'deploy.prototxt'
weights_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
face_detect = cv2.dnn.readNet(prototxt_path, weights_path)

# CNN
if arm:
    from tflite_runtime.interpreter import Interpreter
    model = Interpreter('model.tflite')
    model.allocate_tensors()
else: 
    from tensorflow.keras.models import load_model
    model_folder = 'model'
    model = load_model(model_folder)

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
        except ValueError:
            pass

# Image properties
image_size = 224

unique_labels = ['no mask', 'mask']

while True:
    # Simulate a system interaction
    input('Press Enter to start.')

    # Camera init
    video_capture = cv2.VideoCapture(0)

    detected = 0

    results_mask = np.array([]).astype(np.int32)
    if arm:
        results_temp = np.array([])

    # Set amount of detection and has be an odd number
    while detected < 7:
        # Read frame from camera
        ret, frame = video_capture.read()
        if not ret:
            break
        h, w = frame.shape[:2]

        if arm:
            # Setup array for sorting all 768 temperatures
            mlx_frame = np.zeros((24*32,))
            try:
                mlx.getFrame(mlx_frame)
            # If there is a bad physical connection to the thermal camera
            except OSError:
                pass

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
            # t_frame_d = np.stack((t_frame_d,)*3, axis=-1)

            # Flip camera frame
            frame = cv2.flip(frame, -1)
            # Crop and resize camera frame to try and match thermal frame
            frame = cv2.resize(frame[20:400, 40:527], (640,480))

       # Display the resulting frame
        cv2.imshow('window', frame)
        cv2.waitKey(1)

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
                # A face is found, add 1 to detected
                detected += 1

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

                # Make a prediction
                if arm:
                    pred = classify_image(model, frame_crop_resize_reshape)
                else:
                    pred = model.predict(frame_crop_resize_reshape)

                # Convert prediction to lables
                pred_index = np.argmax(pred[0])
                # Append the prediction to the results
                results_mask = np.append(results_mask, pred_index)

                if arm:
                    # Crop thermal frame of face
                    t_frame_d_crop = t_frame[startY:endY, startX:endX]
                    # Calculate mean temperature of cropped thermal frame
                    # temp = np.round(np.mean(t_frame_d_crop), 1)
                    temp = t_frame_d_crop[round(t_frame_d_crop.shape[0] / 2)][round(t_frame_d_crop.shape[1] / 2)]
                    # Append the temperature to the results
                    results_temp = np.append(results_temp, temp)

    video_capture.release()
    cv2.destroyAllWindows()

    # Get the most common prediction
    results_mask = np.bincount(results_mask).argmax()
    if arm:
        results_temp = results_temp.mean()
    
    print(f'Results show {unique_labels[results_mask]}' + '{0}'.format(f', and your temperature is {np.round(results_temp, 1)}C' if arm else '.'))