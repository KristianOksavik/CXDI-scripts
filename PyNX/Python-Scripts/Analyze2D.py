#!/usr/bin/env python3
# coding: utf-8

import os
import sys
from math import ceil, floor
import numpy
import pyopencl
from pyopencl import array as cla
import time
import glob
import fabio
import h5py
import hdf5plugin
from silx.opencl.processing import OpenclProcessing, BufferDescription, KernelContainer
from silx.opencl.common import query_kernel_info
from pynx.cdi.cdi import save_cdi_data_cxi
import codecs
import argparse
import cdi_tools 

def as_str(smth):
    "Ensure to be a string"
    if isinstance(smth, bytes):
        return smth.decode()
    else:
        return str(smth)

def expand_args(args):
    """
    Takes an argv and expand it (under Windows, cmd does not convert *.tif into
    a list of files.

    :param list args: list of files or wildcards
    :return: list of actual args
    """
    new = []
    for afile in args:
        if glob.has_magic(afile):
            new += glob.glob(afile)
        else:
            new.append(afile)
    return new
    
def parse():

    epilog = """Assumption: There is enough memory to hold all frames in memory
     
                return codes: 0 means a success. 1 means the conversion
                contains a failure, 2 means there was an error in the
                arguments"""

    parser = argparse.ArgumentParser(prog="Analyze2D",
                                     description=__doc__,
                                     epilog=epilog)
    parser.add_argument("IMAGE", nargs="*",
                        help="file with input images in Bliss format HDF5")

    group = parser.add_argument_group("main arguments")
    
    group.add_argument("-o", "--output", default='reciprocal_volume.cxi', type=str,
                       help="output filename in CXI format")
    group.add_argument("-s", "--shape", default=1024, type=int,
                       help="Size of the reciprocal volume, by default 1024³")
    group.add_argument("--scale", default=1.0, type=float,
                       help="Scale (down) the voxel coordinates. For example a factor 2 is similar to a 2x2x2 binning of the volume")

    group.add_argument("-m", "--mask", dest="mask", type=str, default=None,
                       help="Path for the mask file containing both invalid pixels and beam-stop shadow")

    group = parser.add_argument_group("optional behaviour arguments")
    
    group.add_argument("--dry-run", dest="dry_run", action="store_true", default=False,
                       help="do everything except modifying the file system")
    group.add_argument("--profile", action="store_true", default=False,
                       help="Turn on the profiler and print OpenCL profiling at output")
    group.add_argument("--maxi", default=None, type=int,
                       help="Limit the processing to a given number of frames")

    group = parser.add_argument_group("Experimental setup options")
    
    group.add_argument("-d", "--distance", type=float, default=None,
                       help="Detector distance in meter")
    group.add_argument("-b", "--beam", nargs=2, type=float, default=None,
                       help="Direct beam in pixels x, y, by default, the center of the image")
    group.add_argument("-p", "--pixelsize", type=float, default=75e-6,
                       help="pixel size, by default 75µm")

    group = parser.add_argument_group("Scan setup")
    
    group.add_argument("--rot", type=str, default="ths",
                       help="Name of the rotation motor")
    group.add_argument("--scan", type=str, default="scan sz",
                       help="Name of the rotation motor")
    group.add_argument("--scan-len", type=str, dest="scan_len", default="1",
                       help="Pick scan which match that length (unless take all scans")
    group = parser.add_argument_group("Oversampling options to reduces the moiré pattern")
    group.add_argument("--oversampling-img", type=int, dest="oversampling_img", default=7,
                       help="How many sub-pixel there are in one pixel (squared)")
    group.add_argument("--oversampling-rot", type=int, dest="oversampling_rot", default=1,
                       help="How many times a frame is projected")
    group = parser.add_argument_group("OpenCL options")
    group.add_argument("--device", type=int, default=None, nargs=2,
                       help="Platform and device ids")
    try:
        args = parser.parse_args()

        if len(args.IMAGE) == 0:
            raise argparse.ArgumentError(None, "No input file specified.")

        # the upper case IMAGE is used for the --help auto-documentation
        args.images = expand_args(args.IMAGE)
        args.images.sort()
    except argparse.ArgumentError as e:
        logger.error(e.message)
        logger.debug("Backtrace", exc_info=True)
        return EXIT_ARGUMENT_FAILURE
    else:
        return args



def readh5(data, datashape):
    result = 0
    with h5py.File(data, mode="r") as h5:

        for entry in h5.values():
            for entry1 in entry.values():
                if as_str(entry1.attrs.get("NX_class")) == "NXcollection":
                    for dataset in entry1.values():
                        result = dataset[()]
    return result
    
def crop_center(img,center,shape):
    """
    https://stackoverflow.com/questions/43463523/center-crop-a-numpy-array
    user zbw, 2017
    """
    cropx = cropy = shape
    c,y,x = img.shape
    startx = int(center[1] - cropx//2)
    starty = int(center[0] - cropy//2)    
    return img[:, starty:int(starty+cropy), startx:int(startx+cropx)]

def main():
    """Main program
    
    :return: exit code
    """
    config = parse()
    if isinstance(config, int):
        return config

    if len(config.images) == 0:
        raise RuntimeError("No input file provided !")
    
    #frames = {}
    
    if config.mask != None:
        mask = fabio.open(config.mask).data
    else:
        mask = fabio.open("eiger4m_mask_full.npy").data
    
    mask = mask[None, :, :]
    
    t0 = time.perf_counter()
    
    tmp_mon = {}
    shape = (50, 2162, 2068)
    
    
    
    
    #for fn in config.images:
    #    tmp = readh5(fn, shape)
    #    for i, frame in enumerate(tmp):
    #        frames.update({i:frame})
    #    #tmp_mon.update(tmp[1])
    
    frames = readh5(config.images[0], shape)
    #frames = numpy.load(config.images[0])
    
    if frames.ndim == 2:
        frames = numpy.expand_dims(frames, 0)
    
    for i, frame in enumerate(frames):
        frames[i] = numpy.where(mask>0, 0, frame) 
    
    if config.beam is not None:
        center = (config.beam[1], config.beam[0])
        frames = crop_center(frames, center, config.shape)
    
    if len(frames) == 0:
        raise RuntimeError("No valid images found in input file ! Check parameters `--rot`, `--scan` and `--scan-len`")
        
    t2 = time.perf_counter()
    if not config.dry_run:
        if config.output.endswith(".npy"):
            numpy.save(config.output, frames) #frames
        else:
            save_cxi(frames, config, mask=mask)
    
    
    
    return 0
        
def save_cxi(data, config, mask=None):
    save_cdi_data_cxi(config.output, data,
                      wavelength=None,
                      detector_distance=config.distance,
                      pixel_size_detector=config.pixelsize,
                      mask=mask,
                      sample_name=None,
                      experiment_id=None,
                      instrument=None,
                      note=None,
                      iobs_is_fft_shifted=False,
                      process_parameters=None)


if __name__ == "__main__":
    result = main()
    sys.exit(result)

