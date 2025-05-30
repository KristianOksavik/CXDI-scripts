#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import os
import sys

currentpath = os.getcwd()
filename = "logo_7um_0001"
file = f"{currentpath}/ESRF/{filename}/{filename}_1728_scan28_auto_clean.npy"
#file = "/home/kristianos/ESRF/support_apertures_0001_split_026-2024-09-12T11-modes.npy"
#destination = filename
destination = f"{currentpath}/ESRF/{filename}/{filename}_scan28"

def npysplit(file, all_files=True):
    """
    file: The .npy file to split up along x axis (x, y, z)
    """
    array = np.load(file)
    if all_files:
        for i, frame in enumerate(array):
            np.save(f"{destination}/{filename}_split_{i+1:03d}.npy", frame)
    else:
        np.save(f"{destination}/{filename}_split_01.npy", array[0])
    return 0

def main():
    npysplit(file, all_files=True)
    return 0
    
if __name__ == "__main__":
    result = main()
    if result == 0:
        print("Success") 