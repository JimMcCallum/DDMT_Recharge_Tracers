# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 08:28:37 2021

@author: 00098687
"""
import numpy as np
import scipy as sp

def model(R,zmax,alpha,L,theta,beta):
    '''
    This is the main model function. 

    Parameters
    ----------
    R : float
        The recharge rate to the system (meters/year).
    zmax : float
        Depth of lower aquitard (meters).
    alpha : float
        longitudinal dispersivity (meters)
    L : float
        Thickness of immobile diffusion zone (meters).
    theta : float
        valye of the mobile zone potosity (-).
    beta : float
        ratio of mobile to immobile porosity.

    Returns
    -------
    z: An array of discrete depth values that the concentrations were calculated for (m)
    
    c1: Discrete values of carbon-14 activities (pmC) the fires 1:nz values are for the mobile zone. The nz: values are for the immobile zone.
    
    c2: As above but for CFC12 (pptv)

    '''
    dz = 1.
    z = np.arange(0.,zmax+0.1,dz)
    nn = len(z)
    ne = nn-1
    
    #Import the concentrations
    a = np.loadtxt('atmos.dat',skiprows = 1)
    v = vogel_velocity(zmax,z,R,theta)[::-1]
    v = (v[1:]+v[:-1])/2.
    
    #set dispersion parameterrd
    D1 = v*alpha + 3.65e-2
    lam = np.log(2.)/5730.
    D2 = v*alpha + 3.41e-2
    alp1 = 3.65e-2/L**2.
    alp2 = 3.41e-2/L**2.
    # construct equations.
    A1 = np.zeros((nn*2,nn*2))
    A2 = np.zeros((nn*2,nn*2))
    C = np.zeros((nn*2,nn*2))
    RHS1 = np.zeros(nn*2)
    RHS2 = np.zeros(nn*2)
    for i in range(ne):
        Ae,Ce, Aexch = ADE1DEL(v[i],D1[i],dz,lam,alp1,beta)
        A1[i,i] += Ae[0,0] 
        A1[i,i+1] += Ae[0,1]
        A1[i+1,i] += Ae[1,0]
        A1[i+1,i+1] += Ae[1,1] 
        A1[i,i] += Aexch[0,0]  
        A1[i,i+nn] += Aexch[0,2]
        A1[i+1,i+1] += Aexch[1,1]  
        A1[i+1,i+nn+1] += Aexch[1,2]
        A1[i+nn,i+nn] += Aexch[2,2]  
        A1[i+nn,i] += Aexch[2,0]
        A1[i+1+nn,i+1+nn] += Aexch[3,3]  
        A1[i+1+nn,i+nn] += Aexch[3,1]    
        C[i,i] += Ce[0,0]
        C[i+1,i+1] += Ce[1,1]
        C[i+nn,i+nn] += Ce[2,2]
        C[i+1+nn,i+1+nn] += Ce[3,3]
        Ae,Ce, Aexch = ADE1DEL(v[i],D2[i],dz,0.,alp2,beta)
        A2[i,i] += Ae[0,0]
        A2[i,i+1] += Ae[0,1]
        A2[i+1,i] += Ae[1,0]
        A2[i+1,i+1] += Ae[1,1]
        A2[i,i] += Aexch[0,0]  
        A2[i,i+nn] += Aexch[0,2]
        A2[i+1,i+1] += Aexch[1,1]  
        A2[i+1,i+nn+1] += Aexch[1,2]
        A2[i+nn,i+nn] += Aexch[2,2]  
        A2[i+nn,i] += Aexch[2,0]
        A2[i+1+nn,i+1+nn] += Aexch[3,3]  
        A2[i+1+nn,i+nn] += Aexch[3,1]        
    
    A1[0,0] -= 10000.    
    RHS1[0] = -10000. * 100.
    A2[0,0] -= 10000.  
    c1 = np.linalg.solve(A1,RHS1)
    #print(c1)
    c2 = np.zeros(nn*2)
    dt = 5.
    t = 1930.
    ## SS fot initial
    while t < 2020.:
        n = np.argmin((a[:,0]-t)**2.)
        RHS1[0] = -10000. * a[n,1]
        RHS2[0] = -10000. * a[n,4]
        c1 = np.linalg.solve(A1 + C/dt,RHS1 + np.dot(C/dt,c1))
        c2 = np.linalg.solve(A2 + C/dt,RHS2 + np.dot(C/dt,c2))
        t+=dt
    return(z,c1,c2)
    

def vogel_velocity(H,z,R,theta):
    '''
    calculates the velocity depth prfiles using the equations of Vogel (1967).
    
    Refs:
        Vogel, J.C., 1967. Investigation of groundwater flow with radiocarbon. 
        IAEA, International Atomic Energy Agency (IAEA). 

    Parameters
    ----------
    H : float
       aquifer thickness (m).
    z : float
        array of discrete depth values where the velocity will be calculated (m).
    R : float
        recharge rate (meters per year).
    theta :float
        porosity (of the mobilecomponent in this case) .

    Returns
    -------
    Velocity at the dicrete depth intervals in an array (m/year).

    '''
    return(R/theta*z/H)


def ADE1DEL(v,D,dx,lam=0.,alpha = 0.,beta = 0.):
    '''
    This is the element weighting function for the indivisual elements.
    

    Parameters
    ----------
    v : float
        elemental velocity (m/year)
    D : float
       elemental; dispersion coefficient (m.m/year)
    dx : float
        length of elemenst
    lam : float, optional
        elemental decay coefficient (/year). The default is 0..
    alpha : float, optional
        elemental exchange coefficient (/year). The default is 0..
    beta : float, optional
        elemental ratio of mobile to immobile porosity. The default is 0..

    Returns
    -------
    elemental conductance, capacietence and echange matrices.

    '''
    
    
    A = D /dx * np.array([[1.,-1.],[-1.,1.]]) + v / 2. * np.array([[-1.,1.],[-1.,1.]])
    A += lam *dx/2 * np.array([[1.,0.],[0.,1.]])
    C = dx  / 2. * np.array([[1.,0.,0.,0.],
                             [0.,1.,0.,0.],
                             [0.,0.,beta,0.],
                             [0.,0.,0.,beta]])
    Aexch = dx/2.*np.array([[alpha,0.,-alpha,0.],
                            [0.,alpha,0.,-alpha],
                            [-alpha,0.,alpha,0.],
                            [0.,-alpha,0.,alpha]])
    return(A,C,Aexch)




