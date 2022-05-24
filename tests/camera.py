import numpy as np
import cv2
import time

def cycle_calc(time_lst):
    for i in range(len(time_lst) - 1):
        time_lst[i] = time_lst[i + 1] - time_lst[i]
    time_lst.pop(-1)
    return time_lst

cap = cv2.VideoCapture(0)

cycle_time = []

time1 = 0
time2 = 0

while True:
    _, frame = cap.read()

    frame = cv2.flip(frame, -1)

    time1 = time.time()
    cycle_time.append(time1)
    fps = 1 / (time1 - time2)
    time2 = time1
    
    cv2.putText(frame, str(int(fps)) + 'fps', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
    
    cv2.imshow('window', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


mean_cycles = np.round(np.mean(cycle_calc(cycle_time)), 6) * 1000

print(mean_cycles)

cap.release()
cv2.destroyAllWindows()