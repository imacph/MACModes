import numpy as np
from scipy import sparse as sp
from sympy.calculus import finite_diff_weights as fDWeights
import matplotlib.pyplot as plt

def calcRadialGrid(aspectRatio,nPoints):
    # chebyshev grid in [Rmin, Rmax]

    rMax = 1.0 / ( 1 - aspectRatio )
    rMin = aspectRatio * rMax

    return (np.cos(np.linspace(0,np.pi,nPoints))[::-1] + (rMax + rMin) ) * 0.5

def calcSecOrdDerivMats(rGrid,nPoints):
    secOrdD1 = sp.lil_matrix((nPoints,nPoints))
    secOrdD2 = sp.lil_matrix((nPoints,nPoints))

    for i in range(1,nPoints-1):

        weights = fDWeights(2,rGrid[i-1:i+2],rGrid[i])
        
        secOrdD1[i,i-1:i+2] = weights[1][-1]
        secOrdD2[i,i-1:i+2] = weights[2][-1]
    
    return secOrdD1, secOrdD2

def calcForOrdDerivMats(rGrid,nPoints):
    forOrdD1 = sp.lil_matrix((nPoints,nPoints))
    forOrdD2 = sp.lil_matrix((nPoints,nPoints))
    forOrdD4 = sp.lil_matrix((nPoints,nPoints))

    for i in range(2,nPoints-2):

        weights = fDWeights(4,rGrid[i-2:i+3],rGrid[i])
        
        forOrdD1[i,i-2:i+3] = weights[1][-1]
        forOrdD2[i,i-2:i+3] = weights[2][-1]
        forOrdD4[i,i-2:i+3] = weights[4][-1]
    
    return forOrdD1, forOrdD2, forOrdD4

def calcTorProjMat(rGrid,nPoints):
    ''' Construct toroidal projection matrix (no tangential stress) '''
    # in theory this projection should enforce dZ/dr - 2Z/r = 0 at r = rMin and r = rMax
    # the projection matrix P, acts on the interior points of Z to give the full Z vector including boundary points that satisfy the BCs

    toroidalProjMat = sp.lil_matrix((nPoints,nPoints-2))

    edgeWeights = fDWeights(1, rGrid[0:3], rGrid[0])[1][-1]

    toroidalProjMat[0,0] = edgeWeights[1]/(2-edgeWeights[0]*rGrid[0]) * rGrid[0]
    toroidalProjMat[0,1] = edgeWeights[2]/(2-edgeWeights[0]*rGrid[0]) * rGrid[0]

    edgeWeights = fDWeights(1, rGrid[-3:], rGrid[-1])[1][-1]

    toroidalProjMat[-1,-1] = edgeWeights[0]/(2-edgeWeights[2]*rGrid[-1]) * rGrid[-1]
    toroidalProjMat[-1,-2] = edgeWeights[1]/(2-edgeWeights[2]*rGrid[-1]) * rGrid[-1]

    for i in range(1,nPoints-1):
        toroidalProjMat[i,i-1] = 1.

    return toroidalProjMat

def calcPolProjMat(rGrid,nPoints):
    ''' Construct poloidal projection matrix (no normal flow + no tangential stress) '''

    poloidalProjMat = sp.lil_matrix((nPoints,nPoints-4))

    edgeWeights = fDWeights(2, rGrid[0:3], rGrid[0])

    poloidalProjMat[1,0] = (edgeWeights[2][-1][-1]+edgeWeights[1][-1][-1])/(2 * edgeWeights[1][-1][1] - edgeWeights[2][-1][1] * rGrid[0]) * rGrid[0]

    edgeWeights = fDWeights(2, rGrid[-3:], rGrid[-1])

    poloidalProjMat[-2,-1] = (edgeWeights[2][-1][0]+edgeWeights[1][-1][0])/(2 * edgeWeights[1][-1][1] - edgeWeights[2][-1][1] * rGrid[-1]) * rGrid[-1]

    for i in range(2,nPoints-2):
        poloidalProjMat[i,i-2] = 1.

    return poloidalProjMat

def degSetup(order,maxDeg):
    
    # generates lists of odd and even spherical harmonic degrees to consider
    # main lists are not including first and last degrees

    # for order = 0 (axisymmetric), min degree is 1
    # for order > 0, min degree is equal to order

    if order == 0:
        
        minDeg = 1
    
    else:
        
        minDeg = order
    
    if minDeg % 2 == 0:
        
        oddDegrees = [i for i in range(minDeg+1,maxDeg,2)]
        evenDegrees = [i for i in range(minDeg+2,maxDeg,2)]
    
    else:
        
        oddDegrees = [i for i in range(minDeg+2,maxDeg,2)]
        evenDegrees = [i for i in range(minDeg+1,maxDeg,2)]
        
    if minDeg % 2 == 0:
        
        evenDegreesFull = [minDeg] + evenDegrees
        oddDegreesFull = oddDegrees
    else:
        
        oddDegreesFull = [minDeg] + oddDegrees
        evenDegreesFull = evenDegrees
        
    if maxDeg % 2 == 0:
        
        evenDegreesFull = evenDegreesFull + [maxDeg]
    
    else:
        
        oddDegreesFull = oddDegreesFull + [maxDeg]
        
    nDegrees = maxDeg - minDeg + 1
        
    return minDeg,nDegrees,oddDegrees,evenDegrees,evenDegreesFull,oddDegreesFull

aspectRatio = 0.35
nPoints = 7

rGrid = calcRadialGrid(aspectRatio,nPoints)

# Derivative matrices (second order for toroidal, fourth order for poloidal)
secOrdD1, secOrdD2 = calcSecOrdDerivMats(rGrid,nPoints)
forOrdD1, forOrdD2, forOrdD4 = calcForOrdDerivMats(rGrid,nPoints)

# Projection matrices 
toroidalProjMat = calcTorProjMat(rGrid,nPoints)
poloidalProjMat = calcPolProjMat(rGrid,nPoints)

torOdd = True # whether toroidal mode is odd or even

maxDeg =5 # maximum spherical harmonic degree to consider (must be odd and >= 3)
azOrd = 0 # azimuthal order

minDeg,nDegrees,oddDegrees,evenDegrees,evenDegreesFull,oddDegreesFull = degSetup(azOrd,maxDeg)

print(oddDegrees,oddDegreesFull)
print(evenDegrees,evenDegreesFull)

Ekman = 1e-4

torIdMat = sp.lil_matrix((nPoints,nPoints))
torIdMat[1:-1,1:-1] = sp.eye(nPoints-2)

torGridMat=sp.lil_matrix((nPoints,nPoints))
torGridMat[1:-1,1:-1] = sp.diags(rGrid[1:-1])

torGridMatInv=sp.lil_matrix((nPoints,nPoints))
torGridMatInv[1:-1,1:-1] = sp.diags(1.0 / rGrid[1:-1])

torGridMatInv2=sp.lil_matrix((nPoints,nPoints))
torGridMatInv2[1:-1,1:-1] = sp.diags(1.0 / (rGrid[1:-1]**2))


polIdMat = sp.lil_matrix((nPoints,nPoints))
polIdMat[2:-2,2:-2] = sp.eye(nPoints-4)

polGridMat = sp.lil_matrix((nPoints,nPoints))
polGridMat[2:-2,2:-2] = sp.diags(rGrid[2:-2])

polGridMatInv = sp.lil_matrix((nPoints,nPoints))
polGridMatInv[2:-2,2:-2] = sp.diags(1.0 / rGrid[2:-2])

polGridMatInv2 = sp.lil_matrix((nPoints,nPoints))
polGridMatInv2[2:-2,2:-2] = sp.diags(1.0 / (rGrid[2:-2]**2))



if torOdd:
    # toroidal modes are odd, ex. (azOrd = 0, maxDeg = 7): Z_1, W_2, Z_3, W_4, Z_5, W_6, Z_7
    #                         ex. (azOrd = 1, maxDeg = 7): Z_1, W_2, Z_3, W_4, Z_5, W_6, Z_7
    #                         ex. (azOrd = 2, maxDeg = 7): W_2, Z_3, W_4, Z_5, W_6, Z_7
    #                         ex. (azOrd = 3, maxDeg = 7): Z_3, W_4, Z_5, W_6, Z_7
    
    matDim = len(oddDegreesFull) * (nPoints-2) + len(evenDegreesFull) * (nPoints-4)

    leftMat = sp.lil_matrix((matDim,matDim))
    rightMat = sp.lil_matrix((matDim,matDim))
    
    deg = minDeg

    if deg % 2 == 1:
        # first degree is odd, toroidal
        diagMat = 2j * azOrd * torIdMat - deg * (deg + 1) * Ekman * (deg*(deg+1)*torGridMatInv2 - secOrdD2)
        upperMat = 2 * deg * (deg + 2) * np.sqrt((deg - azOrd +1)*(deg +azOrd + 1)/((2*deg +1)*(2*deg +3))) * (forOrdD1 + (deg+1)*polGridMatInv )

        leftMat[0:nPoints-2,0:nPoints-2] = toroidalProjMat.T @ diagMat @ toroidalProjMat
        leftMat[0:nPoints-2,nPoints-2:nPoints-2 + nPoints-4] = toroidalProjMat.T @ upperMat @ poloidalProjMat

    if deg % 2 == 0:
        # first degree is even, poloidal
        diagMat = 2j * azOrd * (deg*(deg+1)*polGridMatInv2 - forOrdD2) - deg * (deg + 1) *Ekman * (forOrdD4 - 2 * deg * (deg +1) * polGridMatInv2 @ forOrdD2 + 4*deg*(deg+1)*polGridMatInv @ polGridMatInv2 @ forOrdD1+ deg*(deg+1) * (deg*(deg+1)-6) * polIdMat)
        upperMat = 2 * deg * (deg + 2) * np.sqrt((deg - azOrd +1)*(deg +azOrd + 1)/((2*deg +1)*(2*deg +3))) * (secOrdD1 + (deg+1)*torGridMatInv )

        leftMat[0:nPoints-4,0:nPoints-4] = poloidalProjMat.T @ diagMat @ poloidalProjMat
        leftMat[0:nPoints-4,nPoints-4:nPoints-4 + nPoints-2] = poloidalProjMat.T @ upperMat @ toroidalProjMat

    
    for deg in oddDegrees:

        # toroidal block row
        diagMat = 2j * azOrd * torIdMat - deg * (deg + 1) * Ekman * (deg*(deg+1)*torGridMatInv2 - secOrdD2)
        upperMat = 2 * deg * (deg + 2) * np.sqrt((deg - azOrd +1)*(deg +azOrd + 1)/((2*deg +1)*(2*deg +3))) * (forOrdD1 + (deg+1)*polGridMatInv )
        lowerMat = 2 * (deg -1) * (deg +1) * np.sqrt((deg - azOrd)*(deg + azOrd)/((2*deg -1)*(2*deg +1))) * (forOrdD1 - deg * polGridMatInv )

        # determine row index in leftMat 
        # depends if minDeg is odd or even (if minDeg is even the first degree block is poloidal, shifting all odd degree blocks down by nPoints-4)
        rowStrIdx = (deg - minDeg)//2 * ( (nPoints-2) + (nPoints-4) ) 
        rowStrIdx += ((minDeg+1) % 2) * (nPoints-4) # shift if first block is poloidal
            
        leftMat[rowStrIdx:rowStrIdx + nPoints-2, rowStrIdx:rowStrIdx + nPoints-2] = toroidalProjMat.T @ diagMat @ toroidalProjMat
        leftMat[rowStrIdx:rowStrIdx + nPoints-2, rowStrIdx + nPoints-2:rowStrIdx + nPoints-2 + nPoints-4] = toroidalProjMat.T @ upperMat @ poloidalProjMat
        leftMat[rowStrIdx:rowStrIdx + nPoints-2, rowStrIdx - (nPoints - 4):rowStrIdx] = toroidalProjMat.T @ lowerMat @ poloidalProjMat

    
