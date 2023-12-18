'''
Name:           Jacob Bodera
Date:           September 2023
Description:    Deletes all of the files from a specified directory
                data_capture.py may generate hundreds of images so this script allows for easy deletion
                Please be careful with this file because there are no safeguards if an incorrect file path is specified
'''

import os
import glob

directory = 'camera_images'

files = glob.glob(f'{directory}/*')
for f in files:
    os.remove(f)