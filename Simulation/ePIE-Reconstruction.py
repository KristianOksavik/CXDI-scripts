print("Initializing")
import sys
import numpy
import cupy as np
import progressbar
import matplotlib.pyplot as plt
plt.rcParams['image.cmap']='jet'
from scipy.fft import fftshift, ifftshift
import cupyx.scipy.fft as cufft
from cupyx.scipy.ndimage import zoom
import scipy.fft as xp_fft
from angular_spectrum_method2 import MonochromaticField

real_data = 1
hqprobe = 1
image = "esrf"
beamstop = 1
folder = "thesisFigures/"
dAS = 12.5e-3  # Distance between aperture and sample
alpha=0.8     # T-update
gamma=0.0     # P-update
arraysize=1024 # Size of input arrays
beamwidth=10e-6
wavelength=0.154981e-9
pixel_size=beamwidth/arraysize



def padding(array, scale, vals=0):
    if scale == 1:
        return array
    onelength = (array.shape[0] * scale - array.shape[0])//2
    return np.pad(array, onelength, mode="constant", constant_values=vals)
    
def cropTo(array, length):
    if length == array.shape[0]:
        return array
    halfLength = length//2
    side = array.shape[0]
    return array[side//2-halfLength:side//2+halfLength, side//2-halfLength:side//2+halfLength]

if real_data:
    dp_name = f"/cluster/home/kristaok/reconstruction/ESRF/logo_7um_0001/logo_7um_0001_{arraysize}_auto_clean.npy"
    diff_pattern = np.fft.fftshift(np.load(dp_name)[50])    
    #diff_pattern = np.fft.fftshift(np.load("/cluster/work/kristaok/logo_7um_0001/wavefront/phaseImages/esrf5981FFP/esrf5981_real_FFP_12.5mm.npy"))
    #probe = np.load(f"/cluster/home/kristaok/reconstruction/tv2_6um_aperture_{arraysize}.npy")#[::-1,::-1]
    probe = np.load("/cluster/home/kristaok/reconstruction/6umaperture_10um_1024.npy")
    S = np.where(np.abs(probe)>130, 1, 0)

else:
    shape = "3_sides"
    dp_name = f"/cluster/work/kristaok/logo_7um_0001/wavefront/phaseImages/{image}FFP/{image}_{shape}_FFP_{1000*dAS:.1f}mm.npy"
    #dp_name = f"/cluster/work/kristaok/logo_7um_0001/wavefront/phaseImages/{image}FFP/{image}_{shape}_FFP_1.0mm.npy"
    
    #Simulated probes
    #probe = np.load(f"/cluster/work/kristaok/logo_7um_0001/wavefront/phaseImages/shapedApertures/propagation_{shape}_00.0.npy")
    #S = np.where(probe>0.5, 1, 0)
    
    #When using the reconstructed probe
    probe = np.load(f"/cluster/home/kristaok/reconstruction/tv_6um_aperture_{arraysize}.npy")#[::-1,::-1]
    S = np.where(np.abs(probe)>130, 1, 0)
    diff_pattern = np.fft.fftshift(np.load(dp_name))

skip = 1
try:
    dAS = 1e-5*int(sys.argv[1]) # Used for looping via bash script
except:
    skip = 0
    pass

if probe.ndim == 3:
    probe = probe[0]
if S.ndim == 3:
    S = S[0]
if diff_pattern.ndim == 3:
    diff_pattern = diff_pattern[0]

if beamstop:
    if arraysize == 512:
        beamstopmask = np.fft.fftshift(np.load("/cluster/work/kristaok/logo_7um_0001/wavefront/detector_mask.npy")[256:-256,256:-256])
    if arraysize == 1024:
        beamstopmask = np.fft.fftshift(np.load("/cluster/work/kristaok/logo_7um_0001/wavefront/detector_mask.npy"))
else:
    beamstopmask = np.zeros_like(diff_pattern)

########################################
############    SETUP    ###############
########################################


# Start with obj = y_0 = 0
T = np.ones_like(probe, dtype=np.float32)

# Initialize wavefront class with initial probe
P = probe
w = MonochromaticField(wavelength=wavelength, extent_x=beamwidth, extent_y=beamwidth, Nx=P.shape[0], Ny=P.shape[1], E=P) 
w.propagate(z=dAS)
P = w.get()

# Initial guess
z_0 = T.copy()

diff_pattern = np.abs(diff_pattern) * np.abs(beamstopmask-1)
plt.figure()
plt.imshow(np.log10(np.abs(np.fft.fftshift(diff_pattern))).get())
#plt.show(block=True)

ind = np.where(beamstopmask==1)
temp = P.copy()
w = MonochromaticField(wavelength=wavelength, extent_x=beamwidth, extent_y=beamwidth, Nx=temp.shape[0], Ny=temp.shape[1], E=temp) 

if hqprobe:
    temp = padding(probe, 10)
    v = MonochromaticField(wavelength=wavelength, extent_x=10*beamwidth, extent_y=10*beamwidth, Nx=temp.shape[0], Ny=temp.shape[1], E=temp) 
    v.propagateTSF(z=dAS)
    P = cropTo(v.get(), arraysize)
    del v
del temp
plt.figure()
plt.title("Probe amplitude (arb.)")
plt.imshow(numpy.abs(P.get()))
#np.save("/cluster/work/kristaok/logo_7um_0001/wavefront/phaseImages/shapedApertures/propagation_real_12.5_1024_hq.npy", P.get())

print("Initialization complete, running loop...")

########################################
#############    LOOP    ###############
########################################

iterations = 10000
with progressbar.ProgressBar(max_value=iterations) as p:
    for ic in range(iterations):
        # We start in Fourier space (obj = z_0)
        z_0 = np.fft.fft2( z_0 )
        if not np.all(z_0):
            z_0 += np.ones_like(z_0)*np.finfo(np.float32).eps

        # Apply reciprocal restriction and go back to real space
        z = diff_pattern * z_0 / np.abs(z_0) 
        z[ind] = z_0[ind]
        z = np.fft.ifft2( z )
        z_0 = np.fft.ifft2( z_0 )
            
        # Update transmission and probe functions
        T_1 = T + alpha * np.conjugate(P) / np.power(np.max(np.abs(P)), 2) * (z - z_0)
        P_1 = P + gamma * np.conjugate(T) / np.power(np.max(np.abs(T)), 2) * (z - z_0)
        
        if 1:
            # Backpropagate P_1
            w.set(P_1) 
            w.propagate(z=-dAS)
            
            # Apply real space restrictions to P_1
            P_1 = np.asarray(w.get())
            P_1 = S * np.abs(P_1) * np.where(np.abs(P_1)>=0, 1, 0)
        
            # Propagate P_1
            w.set(P_1)
            w.propagate(z=dAS)
            P_1 = w.get()
                
        # Replace T and P
        T = T_1
        P = P_1        
        z_0 = T*P
                
        p.update(ic)


########################################
########################################
########################################

print("Loop complete")
        
if z_0.ndim == 3:
    z_0 = z_0[0]

T = (T).get()
z_0 = (z_0).get()
P = (P).get()

if skip:
    destination = "thesisFigures/misc/"
    np.save(destination+f"T_propdiff_{1000*dAS:04.2f}mm", T)
    sys.exit()

#plt.figure()
#plt.title("Phase (rad)")
#plt.imshow(numpy.angle(z_0), cmap="Greys")

#plt.figure()
#plt.title("Amplitude (arb.)")
#plt.imshow(numpy.abs(z_0), cmap="Greys")

plt.figure()
plt.title(f"Probe amplitude ({1000*dAS:.1f}mm)")
plt.imshow(numpy.abs(P))
plt.figure()
plt.title(f"Probe phase ({1000*dAS:.1f}mm)")
plt.imshow(numpy.angle(P))

#np.save("P_improvement.npy", P)
plt.figure()
plt.title(f"Transmission function of sample ({1000*dAS:.1f}mm)")
plt.imshow(numpy.abs(T), cmap="Greys")

plt.figure()
plt.title(f"Phase of transmission function ({1000*dAS:.1f}mm)")
plt.imshow(numpy.angle(T), cmap="Greys")
#np.save("experimental_1.25_t.npy", T)
if False:
    if beamstop:
        np.save(f"{image}_t_beamstop_{shape}_{1000*dAS:.1f}mm", T)
    else:
        np.save(f"{image}_t_{shape}_{1000*dAS:.1f}mm", T)

#plt.figure()
#plt.title(f"|T|, {100*dAS:.2f} cm")
#plt.imshow(numpy.abs(T), cmap="Greys")#, vmin=0.8, vmax=1.1)
#plt.savefig(f"tempimages/T_{dAS*100:.2f}.jpg", dpi=500)

plt.show()

save = input("Do you want to save the T-image? y/n ")
if save == "y":
    if real_data:
        np.save(folder+f"real_{1000*dAS:.1f}mm_{iterations}i_{hqprobe}hq_{alpha:.1f}a{gamma:.1f}g_{arraysize}.npy", T)
    else:
        if beamstop:
            np.save(folder+f"{image}_{1000*dAS:.1f}mm_{iterations}i_{alpha:.1f}a{gamma:.1f}g_beamstop_{arraysize}.npy", T)
        else:
            np.save(folder+f"{image}_{1000*dAS:.1f}mm_{iterations}i_{alpha:.1f}a{gamma:.1f}g_{arraysize}.npy", T)
elif save == "n":
    pass
else:
    pass

print("Success")