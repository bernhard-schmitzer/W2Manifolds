import numpy as np
import scipy
import imageio.v3 as imageio

# MultiScaleOT documentation at: https://bernhard-schmitzer.github.io/MultiScaleOT/
# can be installed via pip install MultiScaleOT
import MultiScaleOT


def PCA(dataMat, keep=None):
    """Simple implementation of PCA on a set of vectors, dataMat is expected to contain one vector per row, no centering is performed."""
    nSamples, dim = dataMat.shape
    if dim < nSamples:
        if keep is None:
            keep = dim
        A = dataMat.transpose().dot(dataMat) / nSamples
        eigData = np.linalg.eigh(A)
        eigval = (eigData[0][-keep::])[::-1]
        eigvec = ((eigData[1][:, -keep::]).transpose())[::-1]
    else:
        if keep is None:
            keep = nSamples
        A = dataMat.dot(dataMat.transpose()) / nSamples
        eigData = np.linalg.eigh(A)
        eigval = (eigData[0][-keep::])[::-1]
        eigvec = ((eigData[1][:, -keep::]).transpose())[::-1]

        eigvec = np.einsum(eigvec, [0, 1], dataMat, [1, 2], [0, 2])
        # renormalize
        normList = np.maximum(np.linalg.norm(eigvec, axis=1), 1E-100)
        eigvec = np.einsum(eigvec, [0, 1], 1 / normList, [0], [0, 1])
    return eigval, eigvec

def PCAVector(dataMatVec, mu, keep=None):
    """PCA on list of vector valued functions on a discrete point cloud; one function per row. Shape of dataMatVec is nSamples,nPoints,dim. mu is measure for weighted inner product."""
    nSamples,nPoints,dim=dataMatVec.shape
    muSqrt=mu**0.5
    dataMat=np.einsum(dataMatVec,[0,1,2],muSqrt,[1],[0,1,2]).reshape((nSamples,nPoints*dim))
    eigval,eigvec=PCA(dataMat,keep)
    eigvec=eigvec.reshape((nSamples,-1,dim))
    eigvec=np.einsum(eigvec,[0,1,2],1/muSqrt,[1],[0,1,2])
    return eigval,eigvec

def LogW2(pi, pos0, pos1, baseLog=None):
    """Extract approximate logarithmic map from optimal coupling via
    barycentric projection for W2 metric

    Args:
        pi: optimal coupling in sparse.csr_array format
        pos0: postions of the first marginal masses
        pos1: positions of second marginal masses
        baseLog: logarithmic map on base space (optional, assumes R^d if None is supplied)

    Returns:
        v: approximate tangent vector
        
    [Taken from LinOT library, https://gitlab.gwdg.de/bernhard.schmitzer/linot ]
    """

    if baseLog is None:
        _baseLog = lambda x, y: y - x
    else:
        _baseLog = baseLog

    npts, dim = pos0.shape
    # reserve empty array for vector field
    v = np.zeros((npts, dim), dtype=np.double)

    # go through points in barycenter
    for j in range(npts):
        # check if current row is empty
        if pi.indptr[j + 1] == pi.indptr[j]:
            continue

        # extract masses in that row of the coupling (based on csr format)
        piRow = pi.data[pi.indptr[j]:pi.indptr[j + 1]]
        # normalize masses
        piRow = piRow / np.sum(piRow)
        # extract indices non-zero entries (based on csr format)
        piIndRow = pi.indices[pi.indptr[j]:pi.indptr[j + 1]]

        # need einsum for averaging along first ("zeroth") axis
        v[j, :] = np.einsum(_baseLog(pos0[j], pos1[piIndRow]), [0, 1], piRow, [0], [1])

    return v



def getVecMultiScale(posX,posY,hierarchyDepth,muX=None,muY=None,errorGoal=1E-3,epsIntermed=1.,afterSteps=5,verbose=False):
    """Get approximate transport map from one point cloud to another, via multiscale Sinkhorn + barycentric projection."""
    nX=posX.shape[0]
    nY=posX.shape[0]
    if muX is None:
        _muX=np.ones(nX)
    else:
        _muX=muX
    if muY is None:
        _muY=np.ones(nY)
    else:
        _muY=muY
    posXC=posX.copy()
    posYC=posY.copy()

    MultiScaleSetup1=MultiScaleOT.TMultiScaleSetup(posXC,_muX,hierarchyDepth,childMode=0)
    MultiScaleSetup2=MultiScaleOT.TMultiScaleSetup(posYC,_muY,hierarchyDepth,childMode=0)
    # generate a cost function object
    costFunction=MultiScaleOT.THierarchicalCostFunctionProvider_SquaredEuclidean(
            MultiScaleSetup1,MultiScaleSetup2)
    # eps scaling
    nLayers=MultiScaleSetup1.getNLayers()
    if verbose:
        print([MultiScaleSetup1.getNPoints(i) for i in range(nLayers)])
        print([MultiScaleSetup2.getNPoints(i) for i in range(nLayers)])
    epsScalingHandler=MultiScaleOT.TEpsScalingHandler()
    epsScalingHandler.setupGeometricMultiLayerB(nLayers,epsIntermed,4.,2,afterSteps)

    # error goal
    _errorGoal=errorGoal*np.sum(_muX)
    # sinkhorn solver object
    SinkhornSolver=MultiScaleOT.TSinkhornSolverStandard(epsScalingHandler,
            0,hierarchyDepth,_errorGoal,
            MultiScaleSetup1,MultiScaleSetup2,costFunction
            )

    SinkhornSolver.initialize()
    MultiScaleOT.setVerboseMode(verbose)
    SinkhornSolver.setSafeMode(True)
    SinkhornSolver.setFixDuals(True)
    msg=SinkhornSolver.solve()
    if msg!=0:
        raise ValueError("error during solving:{msg}".format(msg=msg))
    kerneldata=SinkhornSolver.getKernelCSRDataTuple()
    kernel=scipy.sparse.csr_array(kerneldata,shape=(nX,nY))
    w=LogW2(kernel,posX,posY)
    return w

