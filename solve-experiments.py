import sys
from lib.aux import *

print(sys.argv)
tag=sys.argv[1]

dir="data/"

# load problem data
data=np.load(dir+tag+".npz")

# solver parameters
hierarchyDepth=data["param_hierarchyDepth"]
eps=data["param_eps"]
afterSteps=data["param_afterSteps"]

# reference
posX=data["X"]
muX=data["dens"]


# loop through deformed samples
dat=[]
for i in range(len(data["YList"])):
    print(i)
    posY=data["YList"][i]
    muY=data["densList"][i]
    # obtain approximate optimal transport map
    dat.append(getVecMultiScale(posX,posY,hierarchyDepth,muX,muY,errorGoal=1E-3,epsIntermed=eps,afterSteps=afterSteps,verbose=True))

# dump results to file
dat=np.array(dat)
np.savez(dir+tag+"_tan.npz",vec=dat)


