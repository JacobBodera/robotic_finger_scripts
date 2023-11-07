'''
Name:           Jacob Bodera
Created:        September 2023
Description:    This script will connect to the arduino board using pyfirmata and capture pressure sensor data and images.
                Captured images and pressure data points are synced up and images are titled "t-{timestamp}.png"
                The animate() function runs every INTERVAL (ms) and so all real-time logic should be contained within this function
                Outputs pressure data and time signature to output.txt file
'''

from pyfirmata import Arduino, util
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2 as cv
import keyboard
import serial.tools.list_ports

'''    CONSTANTS       '''
SCOPE_LENGTH = 50
INTERVAL = 500
V_SUPPLY = 5.

open("output.txt", "w").close()


'''      FUNCTIONS      '''
def closeCamera(camera):
    del(camera)
    quit()

def findComPort():
    ports = list(serial.tools.list_ports.comports())
    arduinoComports = []
    numPorts = 0
    chosenPort = 0
    for p in ports:
        if 'Arduino' in p[1]:
            arduinoComports.append(p)
            numPorts += 1
    if numPorts == 0:
        print("----- NO COM FOUND -----")
        return 'None'
    elif numPorts == 1:
        print(f"Connecting to {str(arduinoComports[chosenPort][0])}")
        return str(arduinoComports[chosenPort][0])
    else:
        for index in range(len(arduinoComports)):
            print(arduinoComports[index][0])
        port = str(input('Enter the COM port you want to connect to (COM#): '))
        return port

def animate(i):
    # only store data from current value to {scope_length} values before
    x_vals.append(i)
    x_vals.pop(0)
    y_vals.append(analog_input.read()*V_SUPPLY)
    y_vals.pop(0)
    # figure settings
    ax.cla()
    ax.set_ylim(bottom=-0.1, top=1.1, auto=False)
    ax.set_title("Potentiometer Values Over Time")
    fig.set_figwidth(15)
    fig.set_figheight(7)

    current_time = datetime.now()
    difference = current_time - start_time

    with open("output.txt", "a") as f:
        f.write(f"{round(difference.total_seconds(), 2)}, {y_vals[len(y_vals) - 1]}\n")

    value, image = camera.read()
    cv.imwrite('camera_images\\t-'+str(round(difference.total_seconds(), 2))+'.png', image)
    if keyboard.is_pressed('q'):
        closeCamera(camera)

    ax.annotate(text=f"Current Value: {str(y_vals[len(y_vals) - 1])}V", xycoords="figure pixels", xytext=(5, 5), xy=(5, 5))
    ax.annotate(text=f"Time: {round(difference.total_seconds(), 2)}s", xycoords="figure pixels", xytext=(250, 5), xy=(100, 5))
    ax.plot(x_vals,y_vals)


'''    BOARD & CAMERA SETUP     '''
board = Arduino(findComPort())
it = util.Iterator(board)
it.start()

analog_input = board.get_pin('a:0:i')
camera = cv.VideoCapture(1)

x_vals = [0]*(SCOPE_LENGTH)
y_vals = [0]*(SCOPE_LENGTH)

fig, ax = plt.subplots()
start_time = datetime.now()

ani = FuncAnimation(plt.gcf(), animate, interval=INTERVAL, cache_frame_data=False)

plt.tight_layout()
plt.show()

