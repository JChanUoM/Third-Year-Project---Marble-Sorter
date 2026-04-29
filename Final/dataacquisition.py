from time import sleep
from picamera2 import Picamera2
from gpiozero import Servo
import numpy as np
import cv2
import os

# Initialise camera and configure to capture 400x300
picam2 = Picamera2()
config = picam2.create_preview_configuration({"size": (400, 300), "format": "XRGB8888"})
picam2.configure(config)
picam2.start()


# Servo GPIO pin and position variables
servo = Servo(18)
posClose = -0.55
posOpen = 0.2


# Servo move function
def servo_move(position, hold_time = 0.2):
    servo.value = position
    sleep(hold_time)        # allow time for servo to turn fully
    servo.detach()          # detach pin to stop servo jitter
    
servo_move(posClose)        # initialise servo to close position


# placeholder should be changed into category name
category = "placeholder"

# create folder with the category name
os.makedirs(f"dataset/{category}", exist_ok=True)
# exist_ok = True allows the function to be called multiple times without causing error, even if the target already exists

# count will be incremented per new image 
count = 0

print(f"Capturing for: {category}. Press 's' to save, 'q' to quit.")

while True:
    # Capture frame from camera
    frame = picam2.capture_array()
    cv2.imshow("Macro Feed", frame)

    # Wait for user input ('s' to save, 'q' to quit)
    key = cv2.waitKey(1)
    if key == ord('s'):
        # Saves image to category file and count suffix
        img_path = f"dataset/{category}/img_{count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")    
        count += 1              # increment count for next image

        # Push marble to chute opening, allowing next marble to enter evaluation area
        servo_move(posOpen)
        servo_move(posClose)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
