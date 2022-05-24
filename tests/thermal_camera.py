import cv2
import time
import numpy as np
import board, busio
import adafruit_mlx90640

def cycle_calc(time_lst):
    for i in range(len(time_lst) - 1):
        time_lst[i] = time_lst[i + 1] - time_lst[i]
    time_lst.pop(-1)
    return time_lst

possible_thermal_camera_errors = (ValueError, RuntimeError, ZeroDivisionError, OSError)

def tc_bad_connection():
    input('Bad connection to the thermal camera. Try wiggling it and then run the following command: "sudo i2detect -y 1" to see if it\'s connected and then press Enter...')

# Setup I2C
i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)

time1 = 0
time2 = 0

cycle_time = []

# Begin MLX90640 with I2C comm
while True:
    try:
        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
        break
    # If there is a bad physical connection to the thermal camera
    except possible_thermal_camera_errors:
        tc_bad_connection()

while True:
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
    frame = np.reshape(mlx_frame, (24,32))
    t_frame = frame.copy()

    # Flip thermal frame
    frame = np.flip(frame, axis=0)

    frame = (frame - np.min(frame)) / np.ptp(frame)
    
    # Upscale thermal frame to match camera
    frame = frame.repeat(10, axis=0).repeat(10, axis=1)

    time1 = time.time()
    cycle_time.append(time1)
    fps = 1 / (time1 - time2)
    time2 = time1
    
    cv2.putText(frame, str(int(fps)) + 'fps ' + str(np.round(np.max(t_frame), 1)) + 'C', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    
    cv2.imshow('window', frame)
    cv2.imwrite('img.jpg', 255 * frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

mean_cycles = np.round(np.mean(cycle_calc(cycle_time)), 6) * 1000

print('Cycle: ' + str(mean_cycles))

cv2.destroyAllWindows()