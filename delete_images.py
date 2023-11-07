'''
Name:           Jacob Bodera
Date:           September 2023
Description:    Deletes all of the images from in the camera_images directory that comes from data_capture.py
'''

import os
import glob

files = glob.glob('camera_images/*')
for f in files:
    os.remove(f)