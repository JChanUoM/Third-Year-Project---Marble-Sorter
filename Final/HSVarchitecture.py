from gpiozero import Servo, DigitalOutputDevice
from time import sleep, time
from picamera2 import Picamera2
import numpy as np
import cv2

# Initialise camera and configure to capture 400x300
picam2 = Picamera2()
config = picam2.create_preview_configuration({"size": (400, 300), "format": "XRGB8888"})
picam2.configure(config)
picam2.start_preview(True)
picam2.start()
sleep(1)    # Allow camera to stabilise 

# GPIO Pin Configuration
# Using BCM numbering (not physical pin numbers)
servo = Servo(18)                       # SG90 servo pin
LED = DigitalOutputDevice(22)           # white LEDs in light isolation module
# Driver Pins
DIR_PIN = DigitalOutputDevice(20)            
STEP_PIN = DigitalOutputDevice(21)           
EN = DigitalOutputDevice(2)
CFG0 = DigitalOutputDevice(13)
CFG1 = DigitalOutputDevice(19)
CFG2 = DigitalOutputDevice(26)
CFG3 = DigitalOutputDevice(10)

# Initialise driver mode
EN.on()     #active low
CFG0.off()
CFG1.off()  # CFG0 & CFG1 pins are off allowing 8 microstep operation
CFG2.on()  # CFG2 HIGH & CFG3 HIGH allowing 1.7 A RMS with 15 kOhms
CFG3.on()
LED.on()

# Container position for each colour abstracted as integers
containerPosition = {
    "blue": 0,
    "white": 1,
    "green": 2,
    "red": 3,
    "yellow": 4,
    "unknown": 5,
    #add colour
}

# for testing purposes, allowing tally of total colour sorted
total_colour = {
    "blue": 0,
    "white": 0,
    "green": 0,
    "red": 0,
    "yellow": 0,
    "unknown": 0,
}

# SG90 servo position variables
posClose = -0.55
posOpen = 0.2

# Stepper motor operation variables
currentContainer = 0    # current distribution chute position
containerTotal = 7      # total number of sorting bins
motorRange = 235        # full rotation range in degrees
microStep = 8           # driver microstep mode

# How many steps needed to go to adjacent container
# this assumes that the all container entry position are identical
adjacentStep = int((motorRange/containerTotal)/1.8) 


# Servo move function
def servo_move(position, hold_time = 0.2):
    servo.value = position
    sleep(hold_time)        # allow time for servo to turn fully
    servo.detach()          # detach pin to stop servo jitter
    
servo_move(posClose)        # initialise servo to close position


# Evaluate Marble's Color
def evaluate(frame):
    print("evaluating")
    global nextContainer
    
    # Convert to HSV (better for color detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
    # Define color ranges in HSV
    blue_lower = np.array([94,80,2])
    blue_upper = np.array([140, 255, 255])
   
    green_lower = np.array([35, 50, 80])
    green_upper = np.array([90, 255, 255])

    white_lower = np.array([0, 0, 90])
    white_upper = np.array([179, 50, 255])

    red_lower1 = np.array([20, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np. array([179,255,255])

    yellow_lower = np.array([20,100,100])
    yellow_upper = np.array([35,255,255])

   
    # Create masks for each color
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    #add colour
    
    # Bitwise-AND the mask with original frame, effectively segmenting the distinct colours
    blue_result = cv2.bitwise_and(frame, frame, mask=blue_mask)
    green_result =cv2.bitwise_and(frame, frame, mask=green_mask)
    white_result = cv2.bitwise_and(frame, frame, mask=white_mask)
    red_result =cv2.bitwise_and(frame, frame, mask=red_mask)
    yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_mask)
    #add colour

    cv2.imshow("Result - Blue objects", blue_result)
    cv2.imshow("Result - Green objects", green_result)
    cv2.imshow("Results - White objects", white_result)
    cv2.imshow("Results - Red objects", red_result)
    cv2.imshow("Results - Yellow objects", yellow_result)

    cv2.waitKey(5) # allow time for the result to show
    
    # Count non-zero pixels in each mask
    blue_pixels = cv2.countNonZero(blue_mask)
    green_pixels = cv2.countNonZero(green_mask)
    white_pixels = cv2. countNonZero(white_mask)
    red_pixels = cv2. countNonZero(red_mask)
    yellow_pixels = cv2. countNonZero(yellow_mask)
    #add colour

   
    # Determine dominant color
    colors = {
        'blue': blue_pixels,
        'green': green_pixels,
        'white': white_pixels,
        'red': red_pixels,
        'yellow': yellow_pixels,
        # add colour
    }
   
    # Find color with most pixels
    dominant_color = max(colors, key=colors.get)
    
    # A colour is "known" if significant pixels found (20% of image)
    total_pixels = frame.shape[0] * frame.shape[1]
    if colors[dominant_color] < total_pixels * 0.2:
        return "unknown"    # if lower than 20% it is deemed an unknown colour
   
    return dominant_color
    
# Stepper Motor Move Function
def turnStepperMotor(targetContainer):
    global currentContainer
    
    # Using current chute position and target position determine rotation direction
    if targetContainer > currentContainer:
        DIR_PIN.off()	#turn CW
    else:
        DIR_PIN.on()
        
    # Calculate the steps needed for the stepper to turn to correct position
    stepsNeeded = microStep * adjacentStep * abs(targetContainer-currentContainer)
    
    EN.off() #enable driver active low
    
    # Rotate the stepper motor
    for step in range(stepsNeeded):
        STEP_PIN.on()
        sleep(0.001)	#0.5 ms starts to vibrate
        STEP_PIN.off()
        sleep(0.001)
        
    EN.on() #disable driver

    # Using the SG90 servo, push the marble to chute opening and return to original position
    servo_move(posOpen)
    currentContainer = targetContainer; # Update current chute position
    servo_move(posClose)
    
# Sobel Edge Detection
def edgeDetection(frame):
    
    # Load image in grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Sobel operator
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)

    # Convert to uint8
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    
    # Thresholding, if not noise will count as edge pixels
    ret, thresh_grad = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
    
    # Count Edge Pixels
    edge_pixels = cv2.countNonZero(thresh_grad)
    print(f"Edge Pixels: {edge_pixels}")
    
    # Display result
    cv2.imshow("Sobel Edge Detection", gradient_magnitude)

    # If total number of pixels is above 1000, marble is present
    if edge_pixels >= 1000:
        return True
    else: return False

# Main Loop    
while True:
    # Capture frame from camera
    frame = picam2.capture_array()

    # Use edge detection function to detect marble presence
    marble_present = edgeDetection(frame)
    
    # If there is a marble, evaluate the marble category
    if marble_present == True:
        print("Marble Detected")
        dominant_color = evaluate(frame)

        # Print marble colour and add to tally
        print(dominant_color)
        total_colour[dominant_color] += 1
        print(f"Total: {total_colour}")

        # Turn stepper motor to correct bin position
        turnStepperMotor(containerPosition[dominant_color])

    # Wait for user input ('q' to quit)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
    else: print("No Marble Detected")

cap.release()
cv2.destroyAllWindows()


