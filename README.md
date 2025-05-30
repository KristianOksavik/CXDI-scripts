These files are a collection of scripts used for Coherent X-ray Diffraction Imaging analysis.

The code is used for 2D and 3D reconstruction of scans taken by the ESRF, using their python library PyNX.
These were designed to be used with PyNX version 2024.1.X and onwards, due to argument formatting
Also includes python scripts for reconstruction using the ePIE algorithm

Folder contents:

PyNX:
- Bash-Scripts:
  - PyNX-Reconstruction:
      Example file used for reconstruction from the raw data given by the ESRF.
      Walks through the entire process of creating the 3D reciprocal array, reconstruction, and averaging.
  - logos:
      Example file of how the 2D reconstruction process is, requires a bit more manual work to isolate the 2D scans for reconstrucion.
    
- Python-Scripts:
  - Analyze2D and Split2D are used for 2D reconstruction, to first extract the scans from the HDf5 files to .npy, and then splitting them for easier access to each scan.
  - tv_postprocess_yc_new and ..._new2 are used for TV cycles of reconstructed object. Originally created and published by Erik Malm, then modified by me to output a cropped volume.

Simulation:
- Bash-Scripts:
  - propagation:
      Script used for creating reconstructions that iterates over different aperture-to-sample distances
      
- Python-Scripts:
    - ePIE-Reconstruction.py:
      The main code I used for reconstruction of data where an aperture was used in the CXDI setup.
      Contains the processing of a given diffraction pattern and probe and outputs the reconstruction of the object (T) and probe (P)
  - angular_spectrum_method2:
      Code containing the algorithm for propagation used in reconstruction. Created by Rafael de la Fuente

- Diffraction-Simulator:
  The scripts that are used to create far-field diffraction patterns of a given probe and sample
  the "phase" at the end of the scripts can be ignored as they just reference the work with phase-images.
  
  propeprob.py is used first to create propagated versions of the given probe.
  nearfieldpropphase.py creates the transmission function of a given sample by combining it with a propagated probe
  farfieldpropphase.py creates the far-field diffraction patterns of the transmission functions, outputting diffraction patterns that can be reconstructed by ePIE.
