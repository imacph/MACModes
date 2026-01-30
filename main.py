import numpy as np
from scipy import sparse as sp
from sympy.calculus import finite_diff_weights as fDWeights
import matplotlib.pyplot as plt
from scipy.special import lpmv

def trun_fact(l,m):
    # truncated factorial for spherical harmonics normalization
    
    prod = 1

    for k in range(l-m+1,l+m+1):
        
        prod *= k
        
    return prod

def sphrharm(l,m,theta,phi):
    # spherical harmonics degree l order m 
    
    if l < m:
        
        N = 0
    
    if m == 0:
        N = ((2*l+1)/4/np.pi )**(1/2)
    
    elif l >= m:
        N = ((2*l+1)/4/np.pi / trun_fact(l,m))**(1/2)
    
    else:
        N = 0
    
    if m % 2 == 1:
        
        N*=-1 
        # this cancels the Cordon-Shortley 
        # phase factor present by default
        # in scipy assoc. Legendre funcs
    
    return N*lpmv(m,l,np.cos(theta))*np.exp(1j*m*phi)

def visualize_sparse_matrix(matrix, title="Sparse Matrix Structure", 
                            show_singular_rows=True, block_size=None,
                            cmap='viridis', figsize=(12, 10)):
    """
    Visualize a sparse matrix with block structure highlighting.
    
    Parameters:
    -----------
    matrix : scipy.sparse matrix
        The sparse matrix to visualize
    title : str
        Title for the plot
    show_singular_rows : bool
        If True, highlight rows with zero or near-zero norm
    block_size : int or tuple
        Size of blocks for grid overlay. If tuple, (row_block, col_block)
    cmap : str
        Colormap for non-zero values ('viridis', 'RdBu_r', 'coolwarm')
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    # Convert to csr format for efficient row operations
    mat_csr = sp.csr_matrix(matrix)
    m, n = mat_csr.shape
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[20, 1],
                          hspace=0.3, wspace=0.05)
    
    ax_main = fig.add_subplot(gs[0, 0])
    ax_colorbar = fig.add_subplot(gs[0, 1])
    ax_row_info = fig.add_subplot(gs[1, 0])
    
    # Main matrix visualization
    # Get non-zero entries
    coo = mat_csr.tocoo()
    
    # Plot non-zero structure with magnitude coloring
    if len(coo.data) > 0:
        magnitudes = np.abs(coo.data)
        scatter = ax_main.scatter(coo.col, coo.row, 
                                 c=np.log10(magnitudes + 1e-16),
                                 s=100000/max(m*n, 10000), # adaptive point size
                                 marker='s', 
                                 cmap=cmap,
                                 alpha=0.7)
        plt.colorbar(scatter, cax=ax_colorbar, label='log10(|value|)')
    
    ax_main.set_xlim(-0.5, n-0.5)
    ax_main.set_ylim(m-0.5, -0.5)  # Invert y-axis for matrix convention
    ax_main.set_xlabel('Column Index', fontsize=11)
    ax_main.set_ylabel('Row Index', fontsize=11)
    ax_main.set_title(f'{title}\nShape: {m}×{n}, Nonzeros: {mat_csr.nnz} ({100*mat_csr.nnz/(m*n):.2f}%)', 
                     fontsize=12, fontweight='bold')
    ax_main.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add block structure grid if specified
    if block_size is not None:
        if isinstance(block_size, int):
            row_block, col_block = block_size, block_size
        else:
            row_block, col_block = block_size
        
        # Draw block boundaries
        for i in range(0, m+1, row_block):
            ax_main.axhline(y=i-0.5, color='red', linewidth=1.5, alpha=0.6)
        for j in range(0, n+1, col_block):
            ax_main.axvline(x=j-0.5, color='red', linewidth=1.5, alpha=0.6)
    
    # Analyze row properties
    row_norms = np.array([np.linalg.norm(mat_csr.getrow(i).data) 
                          for i in range(m)])
    row_nnz = np.array([mat_csr.getrow(i).nnz for i in range(m)])
    
    # Identify singular/problematic rows
    singular_threshold = 1e-12
    singular_rows = np.where(row_norms < singular_threshold)[0]
    
    if show_singular_rows and len(singular_rows) > 0:
        # Highlight singular rows on main plot
        for row_idx in singular_rows:
            ax_main.axhspan(row_idx-0.5, row_idx+0.5, 
                          color='red', alpha=0.3, zorder=-1)
        
        ax_main.text(0.02, 0.98, f'⚠ {len(singular_rows)} singular rows',
                   transform=ax_main.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='red', alpha=0.3))
    
    # Row statistics subplot
    ax_row_info.bar(range(m), row_nnz, color='steelblue', alpha=0.7, 
                    label='Non-zeros per row')
    if len(singular_rows) > 0:
        ax_row_info.bar(singular_rows, row_nnz[singular_rows], 
                       color='red', alpha=0.7, label='Singular rows')
    ax_row_info.set_xlabel('Row Index', fontsize=10)
    ax_row_info.set_ylabel('# Non-zeros', fontsize=10)
    ax_row_info.set_title('Row Sparsity Pattern', fontsize=10)
    ax_row_info.legend(fontsize=8)
    ax_row_info.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print detailed report
    print("\n" + "="*60)
    print(f"MATRIX ANALYSIS: {title}")
    print("="*60)
    print(f"Shape: {m} × {n}")
    print(f"Total elements: {m*n}")
    print(f"Non-zero elements: {mat_csr.nnz} ({100*mat_csr.nnz/(m*n):.3f}%)")
    print(f"Matrix norm: {sp.linalg.norm(mat_csr):.6e}")
    print(f"\nRow Statistics:")
    print(f"  Min non-zeros per row: {row_nnz.min()}")
    print(f"  Max non-zeros per row: {row_nnz.max()}")
    print(f"  Mean non-zeros per row: {row_nnz.mean():.2f}")
    print(f"  Min row norm: {row_norms.min():.6e}")
    print(f"  Max row norm: {row_norms.max():.6e}")
    
    if len(singular_rows) > 0:
        print(f"\n⚠ WARNING: {len(singular_rows)} SINGULAR/NEAR-ZERO ROWS DETECTED:")
        print(f"  Row indices: {singular_rows.tolist()}")
        print(f"  Row norms: {row_norms[singular_rows]}")
    else:
        print(f"\n✓ No singular rows detected (threshold: {singular_threshold})")
    
    print("="*60 + "\n")
    
    return fig, (ax_main, ax_colorbar, ax_row_info)


def compare_matrices(matrices, labels, block_size=None, figsize=(16, 8)):
    """
    Compare multiple sparse matrices side by side.
    
    Parameters:
    -----------
    matrices : list of scipy.sparse matrices
        List of matrices to compare
    labels : list of str
        Labels for each matrix
    block_size : int or tuple
        Block size for grid overlay
    """
    n_matrices = len(matrices)
    fig, axes = plt.subplots(1, n_matrices, figsize=figsize)
    
    if n_matrices == 1:
        axes = [axes]
    
    for idx, (mat, label) in enumerate(zip(matrices, labels)):
        mat_csr = sp.csr_matrix(mat)
        m, n = mat_csr.shape
        coo = mat_csr.tocoo()
        
        ax = axes[idx]
        
        if len(coo.data) > 0:
            magnitudes = np.abs(coo.data)
            ax.scatter(coo.col, coo.row, 
                      c=np.log10(magnitudes + 1e-16),
                      s=100000/max(m*n, 10000),
                      marker='s', 
                      cmap='viridis',
                      alpha=0.7)
        
        ax.set_xlim(-0.5, n-0.5)
        ax.set_ylim(m-0.5, -0.5)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row' if idx == 0 else '')
        ax.set_title(f'{label}\n{m}×{n}, nnz={mat_csr.nnz}')
        ax.grid(True, alpha=0.3)
        
        # Add block grid
        if block_size is not None:
            if isinstance(block_size, int):
                rb, cb = block_size, block_size
            else:
                rb, cb = block_size
            for i in range(0, m+1, rb):
                ax.axhline(y=i-0.5, color='red', linewidth=1, alpha=0.5)
            for j in range(0, n+1, cb):
                ax.axvline(x=j-0.5, color='red', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    return fig, axes

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
nPoints = 101

rGrid = calcRadialGrid(aspectRatio,nPoints)

# Derivative matrices (second order for toroidal, fourth order for poloidal)
secOrdD1, secOrdD2 = calcSecOrdDerivMats(rGrid,nPoints)
forOrdD1, forOrdD2, forOrdD4 = calcForOrdDerivMats(rGrid,nPoints)

# Projection matrices 
toroidalProjMat = calcTorProjMat(rGrid,nPoints)
poloidalProjMat = calcPolProjMat(rGrid,nPoints)

torOdd = True # whether toroidal mode is odd or even

maxDeg =101 # maximum spherical harmonic degree to consider (must be odd and >= 3)
azOrd = 0 # azimuthal order

minDeg,nDegrees,oddDegrees,evenDegrees,evenDegreesFull,oddDegreesFull = degSetup(azOrd,maxDeg)

#print(oddDegrees,oddDegreesFull)
#print(evenDegrees,evenDegreesFull)

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

    leftMat = sp.lil_matrix((matDim,matDim),dtype=complex)
    rightMat = sp.lil_matrix((matDim,matDim),dtype=complex)
    
    deg = minDeg # first degree block

    if deg % 2 == 1:
        # first degree is odd, toroidal
        diagMat = 2j * azOrd * torIdMat - deg * (deg + 1) * Ekman * (deg*(deg+1)*torGridMatInv2 - secOrdD2)
        upperMat = 2 * deg * (deg + 2) * np.sqrt((deg - azOrd +1)*(deg +azOrd + 1)/((2*deg +1)*(2*deg +3))) * (forOrdD1 + (deg+1)*polGridMatInv )

        # matrix product denoted by @, projection is performed as P^T A P
        leftMat[0:nPoints-2,0:nPoints-2] = toroidalProjMat.T @ diagMat @ toroidalProjMat
        leftMat[0:nPoints-2,nPoints-2:nPoints-2 + nPoints-4] = toroidalProjMat.T @ upperMat @ poloidalProjMat

        rightMat[0:nPoints-2,0:nPoints-2] = toroidalProjMat.T @ (deg*(deg+1)*1j*torIdMat) @ toroidalProjMat

    if deg % 2 == 0:
        # first degree is even, poloidal
        diagMat = 2j * azOrd * (deg*(deg+1)*polGridMatInv2 - forOrdD2) - deg * (deg + 1) *Ekman * (forOrdD4 - 2 * deg * (deg +1) * polGridMatInv2 @ forOrdD2 + 4*deg*(deg+1)*polGridMatInv @ polGridMatInv2 @ forOrdD1+ deg*(deg+1) * (deg*(deg+1)-6) * polIdMat)
        upperMat = 2 * deg * (deg + 2) * np.sqrt((deg - azOrd +1)*(deg +azOrd + 1)/((2*deg +1)*(2*deg +3))) * (secOrdD1 + (deg+1)*torGridMatInv )

        leftMat[0:nPoints-4,0:nPoints-4] = poloidalProjMat.T @ diagMat @ poloidalProjMat
        leftMat[0:nPoints-4,nPoints-4:nPoints-4 + nPoints-2] = poloidalProjMat.T @ upperMat @ toroidalProjMat
        rightMat[0:nPoints-4,0:nPoints-4] = poloidalProjMat.T @ (deg*(deg+1)*1j*(deg*(deg+1)*polGridMatInv2 - forOrdD2)) @ poloidalProjMat

    # inner odd degree blocks
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

        rightMat[rowStrIdx:rowStrIdx + nPoints-2, rowStrIdx:rowStrIdx + nPoints-2] = toroidalProjMat.T @ (deg*(deg+1)*1j*torIdMat) @ toroidalProjMat

    # inner even degree blocks
    for deg in evenDegrees:

        # poloidal block row
        diagMat = 2j * azOrd * (deg*(deg+1)*polGridMatInv2 - forOrdD2) - deg * (deg + 1) *Ekman * (forOrdD4 - 2 * deg * (deg +1) * polGridMatInv2 @ forOrdD2 + 4*deg*(deg+1)*polGridMatInv @ polGridMatInv2 @ forOrdD1+ deg*(deg+1) * (deg*(deg+1)-6) * polIdMat)
        upperMat = 2 * deg * (deg + 2) * np.sqrt((deg - azOrd +1)*(deg +azOrd + 1)/((2*deg +1)*(2*deg +3))) * (secOrdD1 + (deg+1)*torGridMatInv )
        lowerMat = 2 * (deg -1) * (deg +1) * np.sqrt((deg - azOrd)*(deg + azOrd)/((2*deg -1)*(2*deg +1))) * (secOrdD1 - deg * torGridMatInv )

        # determine row index in leftMat
        rowStrIdx = (deg - minDeg)//2 * ( (nPoints-2) + (nPoints-4) )
        rowStrIdx += ( (minDeg) % 2 ) * (nPoints-2) # shift if first block is toroidal

        leftMat[rowStrIdx:rowStrIdx + nPoints-4, rowStrIdx:rowStrIdx + nPoints-4] = poloidalProjMat.T @ diagMat @ poloidalProjMat
        leftMat[rowStrIdx:rowStrIdx + nPoints-4, rowStrIdx + nPoints-4:rowStrIdx + nPoints-4 + nPoints-2] = poloidalProjMat.T @ upperMat @ toroidalProjMat
        leftMat[rowStrIdx:rowStrIdx + nPoints-4, rowStrIdx - (nPoints - 2):rowStrIdx] = poloidalProjMat.T @ lowerMat @ toroidalProjMat

        rightMat[rowStrIdx:rowStrIdx + nPoints-4, rowStrIdx:rowStrIdx + nPoints-4] = poloidalProjMat.T @ (deg*(deg+1)*1j*(deg*(deg+1)*polGridMatInv2 - forOrdD2)) @ poloidalProjMat


    deg = maxDeg # last degree block

    if deg % 2 == 1:

        diagMat = 2j * azOrd * torIdMat - deg * (deg + 1) * Ekman * (deg*(deg+1)*torGridMatInv2 - secOrdD2)
        lowerMat = 2 * (deg -1) * (deg +1) * np.sqrt((deg - azOrd)*(deg + azOrd)/((2*deg -1)*(2*deg +1))) * (forOrdD1 - deg * polGridMatInv )

        rowStrIdx = (deg - minDeg)//2 * ( (nPoints-2) + (nPoints-4) )
        rowStrIdx += ((minDeg+1) % 2) * (nPoints-4) # shift if first block is poloidal

        leftMat[rowStrIdx:rowStrIdx + nPoints-2, rowStrIdx:rowStrIdx + nPoints-2] = toroidalProjMat.T @ diagMat @ toroidalProjMat
        leftMat[rowStrIdx:rowStrIdx + nPoints-2, rowStrIdx - (nPoints - 4):rowStrIdx] = toroidalProjMat.T @ lowerMat @ poloidalProjMat

        rightMat[rowStrIdx:rowStrIdx + nPoints-2, rowStrIdx:rowStrIdx + nPoints-2] = toroidalProjMat.T @ (deg*(deg+1)*1j*torIdMat) @ toroidalProjMat

    if deg % 2 == 0:

        diagMat = 2j * azOrd * (deg*(deg+1)*polGridMatInv2 - forOrdD2) - deg * (deg + 1) *Ekman * (forOrdD4 - 2 * deg * (deg +1) * polGridMatInv2 @ forOrdD2 + 4*deg*(deg+1)*polGridMatInv @ polGridMatInv2 @ forOrdD1+ deg*(deg+1) * (deg*(deg+1)-6) * polIdMat)
        lowerMat = 2 * (deg -1) * (deg +1) * np.sqrt((deg - azOrd)*(deg + azOrd)/((2*deg -1)*(2*deg +1))) * (secOrdD1 - deg * torGridMatInv )

        rowStrIdx = (deg - minDeg)//2 * ( (nPoints-2) + (nPoints-4) )
        rowStrIdx += ( (minDeg) % 2 ) * (nPoints-2) # shift if first block is toroidal

        leftMat[rowStrIdx:rowStrIdx + nPoints-4, rowStrIdx:rowStrIdx + nPoints-4] = poloidalProjMat.T @ diagMat @ poloidalProjMat
        leftMat[rowStrIdx:rowStrIdx + nPoints-4, rowStrIdx - (nPoints - 2):rowStrIdx] = poloidalProjMat.T @ lowerMat @ toroidalProjMat

        rightMat[rowStrIdx:rowStrIdx + nPoints-4, rowStrIdx:rowStrIdx + nPoints-4] = poloidalProjMat.T @ (deg*(deg+1)*1j*(deg*(deg+1)*polGridMatInv2 - forOrdD2)) @ poloidalProjMat



rhsVec = np.zeros(len(oddDegreesFull) * (nPoints-2) + len(evenDegreesFull) * (nPoints-4),dtype=complex)
torGrid = np.zeros(nPoints,dtype=complex)
torGrid[1:-1] = rGrid[1:-1]

rhsVec[0:nPoints-2] = -1j*toroidalProjMat.T @ torGrid  # set first toroidal degree block to 1

freq = np.sqrt(12/7)
lhsMat = freq*rightMat-leftMat

solnVec = sp.linalg.spsolve(lhsMat,rhsVec)

eigVals,eigVecs = sp.linalg.eigs(leftMat.tocsc(),k=25,M=-1j*rightMat.tocsc(),sigma=1j*np.sqrt(12/7),which='SR')
#print(eigVals)
fig,ax = plt.subplots(1,1,figsize=(8,6),dpi=200)

nTheta = 100
thetaGrid = np.cos(np.linspace(1,nTheta,nTheta)*np.pi/(nTheta+1)) * np.pi/2 + np.pi/2

cc= 0
solnVec = eigVecs[:,cc]
# construct solution in space

vPhi = np.zeros( (nPoints, len(thetaGrid)), dtype=complex)
vTheta = np.zeros( (nPoints, len(thetaGrid)), dtype=complex)
vR = np.zeros( (nPoints, len(thetaGrid)), dtype=complex)

for deg in oddDegreesFull:

    rowStrIdx = (deg - minDeg)//2 * ( (nPoints-2) + (nPoints-4) )
    rowStrIdx += ((minDeg+1) % 2) * (nPoints-4) # shift if first block is poloidal

    solnBlock = solnVec[rowStrIdx:rowStrIdx + nPoints-2]

    vR += deg*(deg+1)/rGrid[:,np.newaxis]**2 *(toroidalProjMat @ solnBlock[:,np.newaxis]) * sphrharm(deg,azOrd,thetaGrid[np.newaxis,:],0)
    vTheta += 1j* azOrd* 1/np.sin(thetaGrid[np.newaxis,:]) * 1/rGrid[:,np.newaxis] * (toroidalProjMat @ solnBlock[:,np.newaxis]) * sphrharm(deg,azOrd,thetaGrid[np.newaxis,:],0)
    vPhi += -1/rGrid[:,np.newaxis] * (toroidalProjMat @ solnBlock[:,np.newaxis]) * np.gradient(sphrharm(deg,azOrd,thetaGrid[np.newaxis,:],0),thetaGrid,axis=1)


sGrid = np.array( [rGrid * np.sin(theta) for theta in thetaGrid] ).T
zGrid = np.array( [rGrid * np.cos(theta) for theta in thetaGrid] ).T

ax.contourf(sGrid,zGrid,np.imag(vPhi),levels=50,cmap='RdBu_r')

ax.set_aspect('equal')

fig,ax = plt.subplots(1,1,figsize=(8,6),dpi=200)

ax.scatter(np.real(eigVals),np.imag(eigVals),c='k')
ax.plot(np.real(eigVals[cc]),np.imag(eigVals[cc]),'ro')
plt.show()


'''
# Visualize the matrices
block_size = (nPoints-2, nPoints-2)  # Approximate block size

print("\n" + "="*60)
print("VISUALIZING SYSTEM MATRICES")
print("="*60)

# Visualize left matrix
visualize_sparse_matrix(leftMat, 
                       title="Left Hand Side Matrix (LHS)", 
                       block_size=block_size,
                       cmap='RdBu_r')

# Visualize right matrix  
visualize_sparse_matrix(rightMat, 
                       title="Right Hand Side Matrix (RHS)", 
                       block_size=block_size,
                       cmap='coolwarm')

# Compare side by side
compare_matrices([leftMat, rightMat], 
                ['LHS Matrix', 'RHS Matrix'],
                block_size=block_size)

plt.show()

print("\nComputing eigenvalues...")
print(sp.linalg.eigs(leftMat.tocsc(),k=6,M=rightMat.tocsc(),which='SM')[0])'''