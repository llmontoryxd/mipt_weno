# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 12:30:49 2020

@author: alexc
"""
import numpy as np
from numba import jit, njit
from matplotlib import pyplot as plt
import weno

def minmod(a, b) :
    return np.sign(a+b)*np.minimum(np.abs(a), np.abs(b)) / 2

# @jit
def solver(model, nx, nt, u0_fun, xl = 0., xr = 1.,
           T = 1., bc_type = 'periodic', recon = 0, mmod = 0):
    
    # B.c. are periodic
    
    # allocate arrays
    # solution (integral averages)
    u = np.zeros((nx, model.ne))
    
    # fluxes at all faces
    flux = np.zeros((nx + 1, model.ne))
    # reconstructed values
    # "0" - left value, "1" - right value
    ulr = np.zeros((nx+1, model.ne, 2))
    
    # right-hand side
    rhs = np.zeros((nx, model.ne))
    dx = (xr - xl) / nx
    dt = T / nt
    # cell centers
    xs = np.linspace(xl + dx/2, xr - dx/2, nx)
    
    # set initial condition
    u = u0_fun(xs)

    for it in range(nt):
        # Reconstruction
        for ix in range(1, nx - 1):
            dl =  u[ix, :] - u[ix - 1, :]
            dr  = u[ix + 1, :] - u[ix, :]
            dc = (u[ix + 1, :] - u[ix - 1, :]) / 2
            #ulr[ix, :, 1]     = u[ix, :] - recon * ((1-mmod)*dc + mmod*minmod(dl, dr)) / 2
            ulr[ix, :, 1] = weno.weno_pre()
            ulr[ix + 1, :, 0] = u[ix, :] + recon * ((1-mmod)*dc + mmod*minmod(dl, dr)) / 2

            if model.ne == 3 : 
                if ulr[ix, 0, 1] <= 0 : 
                    ulr[ix, 0, 1] = 0.1
                if ulr[ix, 2, 1] <= 0 :
                    ulr[ix, 2, 1] = 0.1
                if ulr[ix + 1, 0, 0] <= 0 :
                    ulr[ix + 1, 0, 0] = 0.1
                if ulr[ix + 1, 2, 0] <= 0 :
                    ulr[ix + 1, 2, 0] = 0.1
        # Apply boundary conditions
        if (bc_type == 'periodic'):
            dl = u[0, :] - u[-1, :]
            dr = u[1, :] - u[0, :]
            dc = (u[1, :] - u[-1, :]) / 2
            ulr[0, :, 1] = u[0, :] - recon * ((1-mmod)*dc + mmod*minmod(dl, dr)) / 2
            ulr[1, :, 0] = u[0, :] + recon * ((1-mmod)*dc + mmod*minmod(dl, dr)) / 2
            ulr[-1, :, 1] = u[0, :] - recon * ((1-mmod)*dc + mmod*minmod(dl, dr)) / 2

            dl = u[-1, :] - u[-2, :]
            dr = u[0, :] - u[-1, :]
            dc = (u[0, :] - u[-2, :]) / 2
            ulr[-2, :, 1] = u[-1, :] - recon * ((1-mmod)*dc + mmod*minmod(dl, dr)) / 2
            ulr[-1, :, 0] = u[-1, :] + recon * ((1-mmod)*dc + mmod*minmod(dl, dr)) / 2
            ulr[0, :, 0] = u[-1, :] + recon * ((1-mmod)*dc + mmod*minmod(dl, dr)) / 2
        elif (bc_type == 'transparent'):
            ulr[0, :, 1] = u[0, :]
            ulr[1, :, 0] = u[0, :]
            ulr[-2, :, 1] = u[-1, :]
            ulr[-1, :, 0] = u[-1, :]
            ulr[0, :, 0] = ulr[0, :, 1]
            ulr[-1, :, 1] = ulr[-1, :, 0]
        # Compute fluxes
        for jf in range(nx + 1):
            flux[jf, :] = model.flux_rs(ulr[jf, :, 0], ulr[jf, :, 1])
        
        # Compute rhs
        for ix in range(nx):
            rhs[ix, :] = -(flux[ix+1, :] - flux[ix, :]) / dx
        
        # Update u
        u = u + dt * rhs
    return u, xs
    
    

