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

aspectRatio = 0.35
nPoints = 7

rGrid = calcRadialGrid(aspectRatio,nPoints)

secOrdD1, secOrdD2 = calcSecOrdDerivMats(rGrid,nPoints)
forOrdD1, forOrdD2, forOrdD4 = calcForOrdDerivMats(rGrid,nPoints)

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



''' Construct poloidal projection matrix (no normal flow + no tangential stress) '''

poloidalProjMat = sp.lil_matrix((nPoints,nPoints-4))

edgeWeights = fDWeights(2, rGrid[0:3], rGrid[0])

poloidalProjMat[1,0] = (edgeWeights[2][-1][-1]+edgeWeights[1][-1][-1])/(2 * edgeWeights[1][-1][1] - edgeWeights[2][-1][1] * rGrid[0]) * rGrid[0]

edgeWeights = fDWeights(2, rGrid[-3:], rGrid[-1])

poloidalProjMat[-2,-1] = (edgeWeights[2][-1][0]+edgeWeights[1][-1][0])/(2 * edgeWeights[1][-1][1] - edgeWeights[2][-1][1] * rGrid[-1]) * rGrid[-1]

for i in range(2,nPoints-2):
    poloidalProjMat[i,i-2] = 1.

print(poloidalProjMat.toarray())


