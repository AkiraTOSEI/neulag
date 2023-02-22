
'''
This code is based on https://github.com/BensonRen/AEM_DIM_Bench/blob/1ff82bfdcd6b0a736bf184f0bcb8a533743aacbb/Data/Peurifoy/generate_Peurifoy.py
'''
import numpy as np
from scipy.special import jv, yv
import torch
import time
from multiprocessing import Pool
import os
from tqdm import tqdm
def product(a,b):
    z = np.array([a[0]*b[0]+a[2]*b[1],a[1]*b[0]+a[3]*b[1],a[0]*b[2]+a[2]*b[3],a[1]*b[2]+a[3]*b[3]]).transpose()
    return z

def besselj(m,x):
    if np.isnan(np.sqrt(x)).any():
        print("in besselj: your x is nan")
        print(x)
    return jv(m+0.5,x)/np.sqrt(x)

def besseljd(m,x):
    return (m*besselj(m-1,x) - (m+1)*besselj(m+1,x))/(2*m+1)

def bessely(m,x):
    return yv(m+0.5,x)/np.sqrt(x)

def besselyd(m,x):
    return (m*bessely(m-1,x) - (m+1)*bessely(m+1,x))/(2*m+1)

def spherical_TM1(k,l,r_cumel,omega,eps1,eps2):
    k1 = omega*np.sqrt(eps1)
    k2 = omega*np.sqrt(eps2)
    x1 = k1*r_cumel
    x2 = k2*r_cumel

    j1 = besselj(l,x1)
    j1d = besseljd(l,x1)*x1 + j1
    y1 = bessely(l,x1)
    y1d = besselyd(l,x1)*x1 + y1
    j2 = besselj(l,x2)
    j2d = besseljd(l,x2)*x2 + j2
    y2 = bessely(l,x2)
    y2d = besselyd(l,x2)*x2 + y2

    if k == 1: #TE Mode
        M = product([y2d,-j2d,-y2,j2],[j1,j1d,y1,y1d])
    else: #TM Mode
        M = product([eps1*y2d,-eps1*j2d,-y2,j2], [j1,eps2*j1d,y1,eps2*y1d])

    M = [M[:,0], M[:,1], M[:,2], M[:,3]]
    return M

def spherical_TM2(k,l,r,omega,eps):
    cum_r = np.cumsum(r)
    N,K = eps.shape
    K = K - 1
    M = np.empty((N,4))
    M[:,0] = 1
    M[:,1:2] = 0
    M[:,3] = 1
    M = [M[:,0],M[:,1],M[:,2],M[:,3]]

    for i in range(0,K):
        tmp = spherical_TM1(k,l,cum_r[i],omega,eps[:,i],eps[:,i+1])
        tmp = [tmp[0], tmp[1], tmp[2], tmp[3]]
        M = product(tmp,M)
        M = [M[:,0], M[:,1], M[:,2], M[:,3]]
        # product is giving some strange output here, the resulting M is (4,4), but it should be (401,4)

    return M

def spherical_cs(k,l,r,omega,eps):
    M = spherical_TM2(k,l,r,omega,eps)
    tmp = M[0]/M[1]

    R = (tmp - 1j)/(tmp + 1j)
    R = np.expand_dims(R,axis=1)

    coef = (2*l+1)*np.pi/2*(1/np.power(omega,2))*(1/eps[:,-1])
    coef = np.expand_dims(coef,axis=1)

    z = 1-np.power(np.abs(R),2)
    y = np.power(np.abs(1-R),2)

    sigma = np.concatenate((coef,coef),axis=1)*np.concatenate((z,y),axis=1)

    return sigma

def total_cs(r,omega,eps,order):
    sigma = 0
    for o in range(1,order+1):
        sigma = sigma + spherical_cs(1,o,r,omega,eps) + spherical_cs(2,o,r,omega,eps)
    return sigma

def peurifoy_simulate(radii,lamLimit=400,orderLimit=None,epsIn=None):
    if not epsIn:
        lam = np.linspace(lamLimit, 800, (800-lamLimit)+1)
        omega = 2*np.pi/lam

        eps_silica = 2.04 * np.ones(len(omega))
        my_lam = lam/1000
        eps_tio2 = 5.913+(.2441) * 1/(my_lam*my_lam - .0803)
        eps_water = 1.77 * np.ones(len(omega))

        eps = np.empty((len(lam),len(radii)+1))

        for idx in range(len(radii)):
            if idx%2 == 0:
                eps[:,idx] = eps_silica
            else:
                eps[:,idx] = eps_tio2

        eps[:,-1] = eps_water

    else:
        eps = epsIn

    order = 25
    if len(radii) == 2 or len(radii) == 3:
        order = 4
    elif len(radii) == 4 or len(radii) == 5:
        order = 9
    elif len(radii) == 6 or len(radii) == 7:
        order = 12
    elif len(radii) == 8 or len(radii) == 9:
        order = 15
    elif len(radii) == 10 or len(radii) == 11:
        order = 18

    if None != orderLimit:
        order = orderLimit

    spect = total_cs(radii,omega,eps,order)/(np.pi*np.power(np.sum(radii),2))
    processed_spect = spect[0::2,1]

    return processed_spect





def Shell_task_simulator(input: np.array, cfg) -> np.array:
    """
    Stack task (Chen) simulator. This code refers from the code below.
    https://github.com/BensonRen/AEM_DIM_Bench/blob/1ff82bfdcd6b0a736bf184f0bcb8a533743aacbb/utils/helper_functions.py#L196
    Args:
        input(np.array) : input data for the simulator
    Outputs:
        signal(np.array) : output data for the simulator
    """
    assert len(input.shape)==2
    assert input.shape[-1] == 8
    Ypred = []
    Xpred = input * 20. + 50
    for i in tqdm(range(len(Xpred))):
        spec = peurifoy_simulate(Xpred[i, :])
        Ypred.append(spec)
    return np.array(Ypred)
  


