'''
Name:           Jacob Bodera
Date:           September 2023
Desciption:     Connects to board to control pumps
'''

from pyfirmata import Arduino, util
import time
import keyboard

board = Arduino("COM5")

it = util.Iterator(board)
it.start()

# constants
HIGH = 1
LOW = 0

# set up motor pins
in1 = board.digital[5]
in2 = board.digital[6]
enableA = board.digital[7]
in3 = board.digital[4]
in4 = board.digital[3]
enableB = board.digital[2]

# set up forward motor direction
in1.write(HIGH)
in2.write(LOW)
in3.write(HIGH)
in4.write(LOW)

'''
r -> run
h -> half motors
s -> stop
q -> quit
'''
while True:
    if keyboard.is_pressed('r'):
        print('RUNNING')
        enableA.write(HIGH)
        enableB.write(HIGH)
        time.sleep(0.5)
    elif keyboard.is_pressed('h'):
        print('MOTOR A RUNNING & MOTOR B STOPPED')
        enableA.write(HIGH)
        enableB.write(LOW)
        time.sleep(0.5)
    elif keyboard.is_pressed('s'):
        print('STOP')
        enableA.write(LOW)
        enableB.write(LOW)
        time.sleep(0.5)
    elif keyboard.is_pressed('b'):
        print('STOP')
        enableA.write(LOW)
        enableB.write(HIGH)
        time.sleep(0.5)
    elif keyboard.is_pressed('q'):
        break