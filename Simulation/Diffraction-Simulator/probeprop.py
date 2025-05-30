
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import cupy as cu
from scipy.fft import ifft2, fftshift
from scipy.ndimage import rotate
from pynx.wavefront import *
from angular_spectrum_method2 import MonochromaticField

ps = 19.53125e-9#real space pixel size THIS HAS BEEN CHANGED FROM 14.2267e-9
lambda0 = 1.54981e-10 #wavelength
z = 7.05 #propagation distance
beamwidth=10e-6 #matrix width in real length

#datapath = "/cluster/home/kristaok/reconstruction/ESRF/logo_7um_0001/logo_7um_0001_split_050-2024-09-11T15-modes.h5"
#datapath = "/cluster/home/kristaok/reconstruction/ESRF/apertures_0001/recon_scan13/apertures_0001_split_002-2024-09-12T13-25-modes.h5"
#datapath = "/cluster/home/kristaok/reconstruction/ESRF/apertures_0001/apertures_0001_split_026-2024-09-12T12-54-modes.h5"
#datapath = "/cluster/work/kristaok/logo_7um_0001/wavefront/logo_5mm.npy"
#datapath = "/cluster/work/kristaok/logo_7um_0001/wavefront/apertures/phantom_aperture.npy"

sides = 3
sidesarr = [1, 3, 4, 5, 6, 9]
fullstack = True
datapath = f"/cluster/home/kristaok/reconstruction/shapes_and_resolution/polygons/polygon_{sides}_sides_512.npy"
#datapath = "/cluster/home/kristaok/reconstruction/tv2_6um_aperture_512_SCALED.npy"

def readh5(data):
    def as_str(smth):
      "Ensure to be a string"
      if isinstance(smth, bytes):
          return smth.decode()
      else:
          return str(smth)
    result = 0
    with h5py.File(data, mode="r") as h5:
        for entry in h5.values():
            for entry1 in entry.values():
                if as_str(entry1.attrs.get("NX_class")) == "NXdata":
                    for dataset in entry1.values():
                        result = dataset[()] 
                        break   
    return result
    #return result[0,:,:]

def pad(array, size):
        result = array.copy()
        x, y, z = result.shape
        padxl = padxr = (size-x)//2
        padyl = padyr = (size-y)//2
        padzl = padzr = (size-z)//2
        if x+padxl+padxr != size:
            padxr += 1

        if y+padyl+padyr != size:
            padyr += 1

        if z+padzl+padzr != size:
            padzr += 1

        result = np.pad(result, ((0,0), (padyl, padyr), (padzl, padzr)), "constant", constant_values=(False, False))

        return result

#img = readh5(datapath)
#img = np.load(datapath)
#img = img.real
#img = rotate(img, angle=200, axes=(1, 2))[:,20:502,:455]
#img = pad(img, size=1024)
#print(img.shape)

if not fullstack:
    sidesarr = [sides]
    
# Angular Spectrum Method
if True:
    array = [0, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 12.5e-3]
    #array = [1.25e-2]
    for shape in sidesarr:
        img = cu.load(f"/cluster/home/kristaok/reconstruction/shapes_and_resolution/polygons/polygon_{shape}_sides_512.npy")
        #img = np.load(datapath)
        for i, l in enumerate(array):
            w = MonochromaticField(wavelength=lambda0, extent_x=beamwidth, extent_y=beamwidth, Nx=img.shape[-1], Ny=img.shape[-1], E=img) 
            w.propagate(z=l)
            P = w.get().get()
            plt.figure(num=1)
            plt.imshow(np.abs(P), cmap="gray")
            plt.title(f"NFP, {l*1000:.1f} "+r"mm")
            figname = f"shapedAperturesTEST/propagation_{shape}_sides_{l*1000:04.1f}"
            np.save(figname+".npy", P)
            plt.tight_layout()
            plt.savefig(figname+".jpg", dpi=400)
  
# PyNX wavefront
if False:
  array = [0, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 12.5e-3]
  #array = [1.25e-2]
  for shape in sidesarr:
      img = np.load(f"/cluster/home/kristaok/reconstruction/shapes_and_resolution/polygons/polygon_{shape}_sides_512.npy")
      #img = np.load(datapath)
      for i, l in enumerate(array):
        w = Wavefront(img, pixel_size=ps, wavelength=lambda0)
        w.set(img, shift=True)
        w = PropagateNearField(l) * w
        
        w = ImshowAbs(fig_num=1,title=f"NFP, {l*1000:.1f} "+r"mm") * w
        figname = f"shapedApertures/propagation_{shape}_sides_{l*1000:04.1f}"
        #figname = f"shapedApertures/propagation_real2_sides_{l*1000:04.1f}"
        np.save(figname+".npy", w.get(shift=True))
        plt.tight_layout()
        plt.savefig(figname+".jpg", dpi=600)
  
print("success")