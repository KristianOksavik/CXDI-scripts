import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from scipy.fft import fft2, ifft2, fftshift
from pynx.wavefront import *

ps = 9.765625e-9#19.53125e-9#real space pixel size THIS HAS BEEN CHANGED FROM 14.22 nm
lambda0 = 1.54981e-10 #wavelength
z = 7.05 #propagation distance
typeof = "esrf5981"
shape = "real"
theoretical = True
fullstack = False
mask = np.load("/cluster/work/kristaok/logo_7um_0001/wavefront/detector_mask.npy").astype(bool)
savefigs = True
#destination = "/cluster/work/kristaok/logo_7um_0001/wavefront/logo_FFP/"
arr = ["0.0", "0.5", "1.0", "2.0", "3.0", "4.0", "5.0"]
arr = ["12.5"]
arr2 = ["1_sides", "3_sides", "4_sides", "5_sides", "6_sides", "9_sides"]
arr2 = ["real"]

if True:
  if not fullstack:
      arr2 = [shape]
  
  for aperture in arr2:
      for name in arr:
          img = np.load(f"{typeof}NFP/{typeof}_{aperture}_NFP_{name}mm.npy")
          
          if False:
              plt.figure()
              plt.imshow(np.abs(img)[0])
              plt.show()
      
          if True:
              if theoretical:
                  figname = f"{typeof}_{aperture}_FFP_{name}mm"
                  w = np.fft.ifftshift(fft2(img))
                  np.save(f"{typeof}FFP/"+figname+".npy", w)
              else:
                  figname = f"propagation_{typeof}_{name}mm_7.05m"
                  w = Wavefront(img, pixel_size=ps, wavelength=lambda0)
                  w.set(img, shift=True)
                  w = PropagateFarField(z) * w
                  w = ImshowAbs(fig_num=1,title=f"{figname}", cmap="jet") * w
                  np.save(f"{typeof}FFP/"+figname+".npy", w.get(shift=True))
              if False:
                  plt.title(f"{figname}")
                  plt.tight_layout()
                  plt.savefig(f"{typeof}FFP/"+figname+".jpg", dpi=600)
       


print("success")