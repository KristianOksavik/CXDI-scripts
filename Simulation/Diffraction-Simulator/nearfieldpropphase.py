import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase

def samplePen(array, probenum, shape, thickness=216e-9, delta=4.7e-5, beta=3.97e-6):
    aperture = np.load(f"/cluster/work/kristaok/logo_7um_0001/wavefront/phaseImages/shapedApertures/propagation_real_12.5_1024_hq.npy")
    #aperture = np.load(f"/cluster/work/kristaok/logo_7um_0001/wavefront/phaseImages/shapedApertures/propagation_{shape}_0{probenum}_512.npy")
    #aperture = np.load(f"/cluster/work/kristaok/logo_7um_0001/wavefront/phaseImages/shapedAperturesTEST/propagation_{shape}_0{probenum}.npy")
    
    if array.shape != aperture.shape:
        aperture = scipy.ndimage.zoom(aperture, array.shape[0]/aperture.shape[0])
    t = np.exp(2*np.pi/1.55e-10*1j*(delta+1j*beta)*array*thickness)
    t = np.multiply(t, aperture)
    ft = np.fft.fftshift(np.abs(np.fft.fft2(t)))**2
    return t

#arr = ["0.0", "0.5", "1.0", "2.0", "3.0", "4.0", "5.0"]
arr = ["12.5"]
arr2 = ["1_sides", "3_sides", "4_sides", "5_sides", "6_sides", "9_sides"]
filename = "esrf5981"
shape = "real"
fullstack = False
#logo = np.load(f"{filename}PhaseImage.npy")#[::-1,:]
logo = np.load(r"/cluster/home/kristaok/reconstruction/logo_7um_1024_scaled_phaseimg.npy")

destination = f"/cluster/work/kristaok/logo_7um_0001/wavefront/phaseImages/{filename}NFP/" #"/cluster/work/kristaok/logo_7um_0001/wavefront/logo_NFP/"

if False:
    plt.figure()
    plt.imshow(np.log(np.abs(zero)), cmap="jet")
    plt.show()

if not fullstack:
    arr2 = [shape]

if True:
    for aperture in arr2:
        optf = [samplePen(logo, x, aperture) for x in arr]
        np.save(destination+f"{filename}_{aperture}_NFP_12.5mm.npy", optf)
        
if False:
    for aperture in arr2:
        zero, pointfive, one, two, three, four, five = [samplePen(logo, x, aperture) for x in arr]
        np.save(destination+f"{filename}_{aperture}_NFP_0.0mm.npy", zero)
        np.save(destination+f"{filename}_{aperture}_NFP_0.5mm.npy", pointfive)
        np.save(destination+f"{filename}_{aperture}_NFP_1.0mm.npy", one)
        np.save(destination+f"{filename}_{aperture}_NFP_2.0mm.npy", two)
        np.save(destination+f"{filename}_{aperture}_NFP_3.0mm.npy", three)
        np.save(destination+f"{filename}_{aperture}_NFP_4.0mm.npy", four)
        np.save(destination+f"{filename}_{aperture}_NFP_5.0mm.npy", five)

print("Success")