'''
Name:           Jacob Bodera
Date:           September 2023
Desciption:     Connects to board to control new parker pump with PWM
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

highTime = 0.1
lowTime = 1 - highTime

# set up motor pins
controlPin = board.digital[13]

# enter time is seconds
def pwmControl(highTime, lowTime):
    controlPin.write(HIGH)
    time.sleep(highTime)
    controlPin.write(LOW)
    time.sleep(lowTime)


while not keyboard.is_pressed('q'):
    highTime = board.get_pin('a:0:i')
    pwmControl(highTime, lowTime)