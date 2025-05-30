import cupy as np

"""
MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""


"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

This license prohibits others from using the project to promote derived products without written consent. Redistributions, with or without
modification, requires giving appropriate attribution to the author for the original work. Redistributions must:

1. Keep the original copyright on the software
2. Include full text of license inside the software
3. You must put an attribution in all advertising materials

Under the terms of the MPL, it also allows the integration of MPL-licensed code into proprietary codebases, but only on condition those components remain accessible.
It grants liberal copyright and patent licenses allowing for free use, modification, distribution of the work, but does not grant the licensee any rights to a contributor's trademarks.

"""

def scaled_fourier_transform(x, y, U, λ = 1,z =1, scale_factor = 1, mesh = False):
    """ 

    Computes de following integral:

    Uf(x,y) = ∫∫  U(u, v) * exp(-1j*pi/ (z*λ) *(u*x + v*y)) * du*dv

    Given the extent of the input coordinates of (u, v) of U(u, v): extent_u and extent_v respectively,
    Uf(x,y) is evaluated in a scaled coordinate system (x, y) with:

    extent_x = scale_factor*extent_u
    extent_y = scale_factor*extent_v

    """

    Ny,Nx = U.shape    
    
    if mesh == False:
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        xx, yy = np.meshgrid(x, y)
    else:
        dx = x[0,1]-x[0,0]
        dy = y[1,0]-y[0,0]
        xx, yy = x,y

    extent_x = dx*Nx
    extent_y = dy*Ny

    L1 = extent_x
    L2 = extent_x*scale_factor

    f_factor = 1/(λ*z)
    fft_U = np.fft.fftshift(np.fft.fft2(U * np.exp(-1j*np.pi* f_factor*(xx**2 + yy**2) ) * np.exp(1j*np.pi*(L1- L2)/L1 * f_factor*(xx**2 + yy**2 ))))
    
    
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d = dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d = dy))
    fxx, fyy = np.meshgrid(fx, fy)

    Uf = np.fft.ifft2(np.fft.ifftshift( np.exp(- 1j * np.pi / f_factor * L1/L2 * (fxx**2 + fyy**2))  *  fft_U) )
    
    extent_x = extent_x*scale_factor
    extent_y = extent_y*scale_factor

    dx = dx*scale_factor
    dy = dy*scale_factor

    x = x*scale_factor
    y = y*scale_factor

    xx = xx*scale_factor
    yy = yy*scale_factor  

    Uf = L1/L2 * np.exp(-1j *np.pi*f_factor* (xx**2 + yy**2)   - 1j * np.pi*f_factor* (L1-L2)/L2 * (xx**2 + yy**2)) * Uf *1j * (λ*z)

    if mesh == False:
        return x, y, Uf
    else:
        return xx, yy, Uf

def FresnelPropagator(simulation, z):
    # Parameters: E0 - initial complex field in x-y source plane
    #             ps - pixel size in microns
    #             lambda0 - wavelength in nm
    #             z - z-value (distance from sensor to object)
    #             background - optional background image to divide out from
    #               
    E0 = simulation.E
    ps = simulation.extent_x/simulation.Nx
    lambda0 = simulation.λ
    upsample_scale = 1;                 # Scale by which to upsample image
    n = upsample_scale * E0.shape[1] # Image width in pixels (same as height)
    grid_size = ps * n;                 # Grid size in x-direction
   
    
    # Inverse space
    fx = np.linspace(-(n-1)/2*(1/grid_size), (n-1)/2*(1/grid_size), n)
    fy = np.linspace(-(n-1)/2*(1/grid_size), (n-1)/2*(1/grid_size), n)
    Fx, Fy = np.meshgrid(fx, fy)
   
    # Fresnel kernel / point spread function h = H(kx, ky)
    # from Fourier Optics, chapter 4
    #H = np.sqrt(z*lambda0)*np.exp(1j*np.pi*lambda0*z*(Fx**2+Fy**2));
    # sphere=exp(i*k/2/zc*(xx.^2+yy.^2));
    H = np.exp(1j*(2 * np.pi / lambda0) * z) * np.exp(1j * np.pi * lambda0 * z * (Fx**2 + Fy**2))#original
    #H = np.exp(-1j * np.pi * lambda0 * z * (Fx**2 + Fy**2))#Yuriy
    #H = np.exp(2*np.pi*1j*z*np.sqrt((1/lambda0)**2-Fx**2-Fy**2))#Yuriy 2
    #H=cos(pi*lambda0*z*(Fx.^2+Fy.^2)+(2*pi*z)/lambda0)+1i.*sin(pi*lambda0*z*(Fx.^2+Fy.^2)+(2*pi*z)/lambda0);
   
    # Compute FFT centered about 0
    E0fft = np.fft.fftshift(np.fft.fft2(E0))     # Centered about 0 since fx and fy centeredabout 0
   
    # Multiply spectrum with fresnel phase-factor
    G = H * E0fft
    Ef = np.fft.ifft2(G) # Output after deshifting Fourier transform
   
    return Ef

"""
MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

def two_steps_fresnel_method(simulation, E, z, λ, scale_factor):
    """
    Compute the field in distance equal to z with the two step Fresnel propagator, rescaling the field in the new coordinates
    with extent equal to:
    new_extent_x = scale_factor * self.extent_x
    new_extent_y = scale_factor * self.extent_y

    Note that unlike within in the propagate method, Fresnel approximation is used here.
    Reference: VOELZ, D. G. (2011). Computational Fourier optics. Bellingham, Wash, SPIE.

    To arbitrarily choose and zoom in a region of interest, use bluestein method method instead.
    """



    L1 = simulation.extent_x
    L2 = simulation.extent_x*scale_factor


    fft_E = np.fft.fftshift(np.fft.fft2(E * np.exp(1j * np.pi/(z * λ) * (L1-L2)/L1 * (simulation.xx**2 + simulation.yy**2) )  ))
    fx = np.fft.fftshift(np.fft.fftfreq(simulation.Nx, d = simulation.dx))
    fy = np.fft.fftshift(np.fft.fftfreq(simulation.Ny, d = simulation.dy))
    fx, fy = np.meshgrid(fx, fy)

    E = np.fft.ifft2(np.fft.ifftshift( np.exp(- 1j * np.pi * λ * z * L1/L2 * (fx**2 + fy**2))  *  fft_E) )


    simulation.extent_x = simulation.extent_x*scale_factor
    simulation.extent_y = simulation.extent_y*scale_factor

    simulation.dx = simulation.dx*scale_factor
    simulation.dy = simulation.dy*scale_factor

    simulation.x = simulation.x*scale_factor
    simulation.y = simulation.y*scale_factor

    simulation.xx = simulation.xx*scale_factor
    simulation.yy = simulation.yy*scale_factor


    E = L1/L2 * np.exp(1j * 2*np.pi/λ * z   - 1j * np.pi/(z * λ)* (L1-L2)/L2 * (simulation.xx**2 + simulation.yy**2)) * E

    return E

def angular_spectrum_method(simulation, E, z, λ, scale_factor = 1):
    """
    Compute the field in distance equal to z with the angular spectrum method. 
    By default (scale_factor = 1), the ouplut plane coordinates is the same than the input.
    Otherwise, it's recommended to use the two_steps_fresnel_method as it's computationally cheaper.
    To arbitrarily choose and zoom in a region of interest, use bluestein method instead.

    Reference: https://rafael-fuente.github.io/simulating-diffraction-patterns-with-the-angular-spectrum-method-and-python.html
    """
    

    # compute angular spectrum
    fft_c = np.fft.fft2(E)
    c = np.fft.fftshift(fft_c)

    fx = np.fft.fftshift(np.fft.fftfreq(simulation.Nx, d = simulation.dx))
    fy = np.fft.fftshift(np.fft.fftfreq(simulation.Ny, d = simulation.dy))
    fxx, fyy = np.meshgrid(fx, fy)

    argument = (2 * np.pi)**2 * ((1. / λ) ** 2 - fxx ** 2 - fyy ** 2)

    #Calculate the propagating and the evanescent (complex) modes
    tmp = np.sqrt(np.abs(argument))
    kz = np.where(argument >= 0, tmp, 1j*tmp)


    if scale_factor == 1:

        # propagate the angular spectrum a distance z
        E = np.fft.ifft2(np.fft.ifftshift(c * np.exp(1j * kz * z)))

    else:
        nn_, mm_ = np.meshgrid(np.arange(simulation.Nx)-simulation.Nx//2, np.arange(simulation.Ny)-simulation.Ny//2)
        factor = ((simulation.dx *simulation.dy)* np.exp(np.pi*1j * (nn_ + mm_)))


        simulation.x = simulation.x*scale_factor
        simulation.y = simulation.y*scale_factor

        simulation.dx = simulation.dx*scale_factor
        simulation.dy = simulation.dy*scale_factor

        extent_fx = (fx[1]-fx[0])*simulation.Nx
        simulation.xx, simulation.yy, E = scaled_fourier_transform(fxx, fyy, factor*c * np.exp(1j * kz * z),  λ = -1, scale_factor = simulation.extent_x/extent_fx * scale_factor, mesh = True)
        simulation.extent_x = simulation.extent_x*scale_factor
        simulation.extent_y = simulation.extent_y*scale_factor

    return E
    

class MonochromaticField:
    def __init__(self,  wavelength, extent_x, extent_y, Nx, Ny, E):
        """
        Initializes the field, representing the cross-section profile of a plane wave

        Parameters
        ----------
        wavelength: wavelength of the plane wave
        extent_x: length of the rectangular grid 
        extent_y: height of the rectangular grid 
        Nx: horizontal dimension of the grid 
        Ny: vertical dimension of the grid 
        intensity: intensity of the field
        """
        
        self.extent_x = extent_x
        self.extent_y = extent_y

        self.dx = extent_x/Nx
        self.dy = extent_y/Ny

        self.x = self.dx*(np.arange(Nx)-Nx//2)
        self.y = self.dy*(np.arange(Ny)-Ny//2)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.Nx = Nx
        self.Ny = Ny
        self.E = E
        self.λ = wavelength
        self.z = 0
        
    def add(self, optical_element):

        self.E = optical_element.get_E(self.E, self.xx, self.yy, self.λ)
        
    def set(self, E):
    
        self.E = E

    def get(self):
        
        return self.E

    def propagate(self, z, scale_factor = 1):
        """
        Compute the field in distance equal to z with the angular spectrum method
        The ouplut plane coordinates is the same than the input.
        """

        self.z += z
        self.E = angular_spectrum_method(self, self.E, z, self.λ, scale_factor = scale_factor)
        
    def propagateTSF(self, z, scale_factor = 1):
        """
        Compute the field in distance equal to z with the angular spectrum method
        The ouplut plane coordinates is the same than the input.
        """

        self.z += z
        self.E = two_steps_fresnel_method(self, self.E, z, self.λ, scale_factor = scale_factor)
        
    def FresnelPropagator(self, z):
        """
        Compute the field in distance equal to z with the angular spectrum method
        The ouplut plane coordinates is the same than the input.
        """

        self.z += z
        self.E = FresnelPropagator(self, z)
        