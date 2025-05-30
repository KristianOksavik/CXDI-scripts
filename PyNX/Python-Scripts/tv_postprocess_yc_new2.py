#!/usr/bin/env python3
# coding: utf-8


"""
Script for total variation post-processing of CDI reconstructions.
"""

__version__ = "0.1.0"
__author__  = "Erik Malm"
__date__    = "2022.04.26"


import argparse as ap
#import numpy as xp
import numpy as np
import cupy as xp
# to use less memory and run faster YC 05/05/2022
#import scipy.fftpack as xp_fft
import cupyx.scipy.fft as cufft
import scipy.fft as xp_fft
import h5py
import hdf5plugin
from scipy.ndimage import zoom
import os

ww = os.cpu_count()

def _grad_backward_1D(u, ax):
    out = xp.empty_like(u, dtype=u.dtype)
    slice1 = [slice(None)]*u.ndim
    slice2 = [slice(None)]*u.ndim
    slice1[ax] = slice(1, None)
    slice2[ax] = slice(None, -1)
    out[tuple(slice1)] = xp.subtract(u[tuple(slice1)], u[tuple(slice2)])
    slice1[ax] = 0
    slice2[ax] = -1
    out[tuple(slice1)] = out[tuple(slice2)] = 0
    return out

def _grad_forward_1D(u, ax):
    out = xp.empty_like(u, dtype=u.dtype)
    slice1 = [slice(None)]*u.ndim
    slice2 = [slice(None)]*u.ndim
    slice1[ax] = slice(1, None)
    slice2[ax] = slice(None, -1)
    out[tuple(slice2)] = xp.subtract(u[tuple(slice1)], u[tuple(slice2)])
    slice1[ax] = -1
    slice2[ax] = 0
    out[tuple(slice1)] = out[tuple(slice2)] = 0
    return out


def _grad_backward(u):
    outvals = []
    for axis in range(u.ndim):
        out = _grad_backward_1D(u, axis)
        outvals.append(out)
    return xp.array(outvals)

def _grad_forward(u):
    outvals = []
    for axis in range(u.ndim):
        out = _grad_forward_1D(u, axis)            
        outvals.append(out)
    return xp.array(outvals)

def _tv_grad(u,ep=1e-4):
    """ Gradient direction for TV minimization.
    """
    grad_u = xp.array(_grad_backward(u))
    grad_mag = xp.sqrt(xp.sum(xp.square(xp.abs(grad_u)), axis=(0)))
    grad_mag = ep + grad_mag
    tv_grad = xp.zeros_like(u)
    for ax in range(u.ndim):
        tv_grad -= _grad_forward_1D(grad_u[ax]/grad_mag, ax)
    return tv_grad

def _tv_cost(u):
    grad = _grad_forward(u)
    return xp.sum(xp.sqrt(xp.abs(grad * grad.conj())))

def slice_center(arr, new_shape):
    """
    Slice a numpy array from its center to the specified new shape.

    Parameters:
    - arr: The input numpy array (must be 3-dimensional).
    - new_shape: A tuple (x, y) specifying the size of the new slice.

    Returns:
    - A numpy array sliced from the center of the original array.
    """
    # Calculate the center of the original array
    center = np.array(arr.shape) // 2

    # Calculate the start and end indices for each dimension
    start = center - np.array(new_shape) // 2
    end = center + np.array(new_shape) // 2

    # Slice and return the array from the center
    return arr[start[0]:end[0], start[1]:end[1]]

def solve_tv_problem(u, b, mask, stepsize, nits, verbose):
    """Total variation minimization
    u: numpy array fftshifted into corners
    mask: boolean mask fftshift into corners: 0: no data; 1: data
    stepsize: step size ( 1e-3 - 1e-2)
    nits: number of iterations
    verbose: print iteration number.
    """
    sc = 1.0/xp.max(xp.abs(u))
    u, b = sc*u, sc*b
    costs = []
    for a in range(nits):
        u = u - stepsize * xp_fft.fftshift(_tv_grad(xp_fft.ifftshift(u)))
        u = xp_fft.fftn(u)#,workers=ww)
        u[mask] = b[mask]
        u = xp_fft.ifftn(u)#,workers=ww)
        if xp.mod(a+1,10)==0 or a==0:
            costs.append(_tv_cost(u))
        if xp.mod(a+1,nits//20)==0 and verbose:
            print("Iteration: %d." %(a+1))
    u, b = u/sc, b/sc
    return u, costs

def parse_inputs():
    """Parse command line arguments"""
    parser = ap.ArgumentParser()
    parser.add_argument("rec_fname", type=str,
                        help="Path of reconstruction: numpy array or cxi data.")
    parser.add_argument("data_fname", type=str,
                        help="Path to data: numpy array.")
    parser.add_argument("-size", type=int, default=10000, dest='dsize',
                        help="Size of the data matrix: int.")
    parser.add_argument("-s", "--stepsize", type=float, default=5e-3, dest="stepsize",
                        help="Step size for TV gradient descent.")
    parser.add_argument("-i", "--iterations", type=int, default=100, dest="nits",
                        help="Number of iterations.")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="Verbose: prints iterations number.")
    parser.add_argument("-r", "--refined", action="store_true", dest="refined",
                        help="Refined: Run a set number of TV iterations for quality.")
    args = parser.parse_args()
    return args


def phasecor(d):
    if (d.real).mean()<0:
        d.real *= -1
    if (d.imag).mean()<0:
        d.imag *= -1
    phi = np.angle((d**2).sum())
    if phi < 0:
        d = (np.real(d)-1j*np.imag(d))
        phi = np.angle((d**2).sum())
    d = d*np.exp(-1j*phi/2)
    return d

'''
def phasecor(d):
    phi = np.angle((d**2).sum())
    if phi < 0:
        d = (np.real(d)-1j*np.imag(d))
        phi = np.angle((d**2).sum())
    d = d*np.exp(-1j*phi/2)
    return d
'''

def h5getarray(filename):
    with h5py.File(filename, mode="r") as h5:
        try:
            data = h5['/entry_1/image_1/data'][0,:,:,:]
            first_mode = 1 #h5['/entry_1/data_2/data'][0]
            command =  "none  none none none none none none" #h5['/entry_1/image_1/process_1/command'][()]
        except:
            #data = h5['/entry_1/image_1/data'][:]
            data = h5['/entry_1/image_1/data'][0,:,:]
            first_mode = 1
            command = "none  none none none none none none"
    #data = data[0,:,:,:]
    print(data.shape)
    return data,first_mode,command

def readnpz(FileName):
    f = np.load(FileName)
    #data = f['data']
    data = f[f.files[0]]
    f.close()
    return data


def load_rec(filename,size):
    """Load reconstruction file"""

    if filename.split(".")[-1]=="cxi":
        data,first_mode,command = h5getarray(filename)
    if filename.split(".")[-1]=="h5":
        data,first_mode,command = h5getarray(filename)
        #print(str(command).split(" ")[-5])
        #print("First mode is %2.2f" % first_mode)
    if filename.split(".")[-1]=="npz":
        data = readnpz(filename)
    if filename.split(".")[-1]=="npy":
        data = np.load(filename)

    #a_angle = np.angle(data[np.abs(data)>100]).mean()
    #data = data*np.exp(-1j*(a_angle-0.785)) 
    #data = phasecor(data)
    shape = data.shape
    if size>np.array(shape).max():
        print("Reconstruction is smaller than array convert to size "+str(size))
        ndata = np.zeros((size,size),data.dtype)
        ndata[size//2-shape[0]//2:size//2-shape[0]//2+shape[0],size//2-shape[1]//2:size//2-shape[1]//2+shape[1]] = data +0
        del data 
        return xp.asarray(ndata), shape
    else:
        return xp.asarray(data), shape


def load_data(filename,size):
    """Load data file and return boolean mask"""
    
    if filename.split(".")[-1]=="npy":
        data = np.load(filename)
    if filename.split(".")[-1]=="edf":    
        data = fabio.open(filename).data
    data[data<0]=0
    
    shape = data.shape
    s1 = shape[0]//2
    s2 = shape[1]//2

    #print("Matrix shape", shape)

    if size<np.array(shape).max():
        print("Diffraction volume is larger than array convert to size "+str(size))
        ndata = np.zeros((size,size),data.dtype)
        ndata = data[shape[0]//2-size//2:shape[0]//2+size//2,shape[1]//2-size//2:shape[1]//2+size//2] + 0
        del data
        ndata[ndata>0] = 1
        ndata[np.isnan(data)]=0
        return xp.asarray(ndata.astype(bool))
    elif size>np.array(shape).max():
        print("Diffraction volume is smaller than array convert to size "+str(size))
        ndata = np.zeros((size,size),data.dtype)
        ndata[size//2-shape[0]//2:size//2+shape[0]//2,size//2-shape[1]//2:size//2+shape[1]//2] = data +0
        del data 
        ndata[ndata>0] = 1
        ndata[np.isnan(data)]=0
        return xp.asarray(ndata.astype(bool))
    else:
        data[data>0]=1
        data[np.isnan(data)]=0
        return xp.asarray(data.astype(bool))

    

def main():
    #Parse input arguments
    args = parse_inputs()

    #Load arrays
    mask = xp_fft.fftshift(load_data(args.data_fname,args.dsize))
    size = mask.shape[0]
    u, orig_shape = load_rec(args.rec_fname,size)
    u = xp_fft.fftshift(u)
        

    stepsize = float(args.stepsize)
    nits = int(args.nits)
    verbose = args.verbose
    refined = args.refined
    
    #Run TV
    with xp_fft.set_backend(cufft):
        if refined:
            u, costs = solve_tv_problem(u, xp_fft.fftn(u), mask, 15*stepsize, 1000, verbose)
            u, costs = solve_tv_problem(u, xp_fft.fftn(u), mask, 5*stepsize, 1000, verbose)
            u, costs = solve_tv_problem(u, xp_fft.fftn(u), mask, 10*stepsize, 500, verbose)
            u, costs = solve_tv_problem(u, xp_fft.fftn(u), mask, 5*stepsize, 500, verbose)
            u, costs = solve_tv_problem(u, xp_fft.fftn(u), mask, 15*stepsize, 1000, verbose)
            u, costs = solve_tv_problem(u, xp_fft.fftn(u), mask, 5*stepsize, 1000, verbose)
            u, costs = solve_tv_problem(u, xp_fft.fftn(u), mask, 10*stepsize, 500, verbose)
            u, costs = solve_tv_problem(u, xp_fft.fftn(u), mask, 5*stepsize, 500, verbose)
            u, costs = solve_tv_problem(u, xp_fft.fftn(u), mask, stepsize, 100, verbose)
        else:
            u, costs = solve_tv_problem(u, xp_fft.fftn(u), mask, 15*stepsize, nits, verbose)
            u, costs = solve_tv_problem(u, xp_fft.fftn(u), mask, stepsize, nits, verbose)

    #Replace high frequency info  
    #ind = np.where(np.abs(u2)>np.abs(u))
    #u2[ind] = u[ind] + 0 

    #Save TV result & costs
    u = slice_center(xp_fft.fftshift(u), orig_shape)
    #u = xp_fft.fftshift(u)
    xp.save("tv_" + args.rec_fname.split("/")[-1], u) 
    #xp.save("tv_costs_" + args.rec_fname.split("/")[-1], xp.array(costs))
    


if __name__ == "__main__":
    main()

