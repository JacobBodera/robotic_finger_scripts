import cv2
import serial.tools.list_ports

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
elif numPorts == 1:
    print(f"Connecting to {str(arduinoComports[chosenPort][0])}")
    print(str(arduinoComports[chosenPort][0]))
else:
    for index in range(len(arduinoComports)):
        print(arduinoComports[index][0])
    port = str(input('Enter the COM port you want to connect to (COM#): '))
    print(port)