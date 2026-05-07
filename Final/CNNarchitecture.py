from gpiozero import Servo, DigitalOutputDevice
from picamera2 import Picamera2
from time import sleep, time
import numpy as np
import cv2

from tensorflow.lite.python.interpreter import Interpreter

# CNN related initialisation
# categories as trained model categories
categories = ["white", "transparent", "none", "blue", "green"]
categories.sort()

# load tflite model, placeholder should be changed to path for tflite model
tfliteModelPath = "placeholder"
interpreter = Interpreter(model_path=tfliteModelPath)

# force input shape to 320x320
input_details = interpreter.get_input_details()
interpreter.resize_tensor_input(input_details[0]['index'], [1,320,320,3])
interpreter.allocate_tensors()

# get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on input data
input_shape = input_details[0]['shape']


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
    "transparent": 6, 
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
    "transparent": 0,
}

# SG90 servo position variables
posClose = -0.55
posOpen = 0.2

# Stepper motor operation variables
currentContainer = 0    # current distribution chute position
containerTotal = 7      # total number of sorting bins
motorRange = 235        # full rotation range in degrees
microStep = 8           # driver microstep mode

# re-adjust Servo variables
marble = True
noneStart = 0

# How many steps needed to go to adjacent container
# this assumes that the all container entry position are identical
adjacentStep = int((motorRange/containerTotal)/1.8) + 1


# Servo move function
def servo_move(position, hold_time = 0.2):
    servo.value = position
    sleep(hold_time)        # allow time for servo to turn fully
    servo.value = None         # detach pin to stop servo jitter
    
servo_move(posClose)        # initialise servo to close position

# Evaluate Marble's Color
def evaluate(frame):
    print("evaluating")
    global nextContainer
    
    #tflite evaluation
    img = cv2.resize(frame, (320,320), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


    input_data = np.array(img, dtype=np.float32)            # float32 [0.0, 1.0]
    input_data = np.expand_dims(input_data, axis=0)         # shape (1, 320, 320, 3)


    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    #interpret output to actual category name
    output_data_tflite =  interpreter.get_tensor(output_details[0]['index'])
    categoryValue = np.argmax(output_data_tflite, axis=1)
    categoryValue = categoryValue[0]

    print(categories[categoryValue])
    return categories[categoryValue]
    
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

# Main Loop    
while True:
    # Capture frame from camera
    frame = picam2.capture_array()

    # CNN evaluation
    predicted = evaluate(frame)

    # If there is marble
    if predicted != "none":
        marble = True
        print(predicted)

        # Add tally to sorted marble
        total_colour[predicted] += 1
        print(f"Total: {total_colour}")
        
        # Turn stepper motor to correct bin position
        turnStepperMotor(containerPosition[predicted])
    
    # If no marble
    else:
        # reset servo position if the system consistently detects no marble for 3 seconds
        # If first time detecting none, set noneStart as current time
        if marble == True:  
            marble = False
            noneStart = time()
            print(f"detected none")
        
        else:
            noneDuration = time() - noneStart
            print(f"none detected for {noneDuration} seconds")
            # if has been detecting none for more than 3 seconds, close servo and restart noneStart
            if noneDuration > 3:
                noneStart = time()
                servo_move(posClose)
        print("none")
        
    # Wait for user input ('q' to quit)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    


    





