import numpy as np
import matplotlib.pyplot as plt

def PlottingPart2D():
    a = np.genfromtxt("CurrentMesh2D.msh",usecols=(0,1,2))

    head = a[0,:]
    nv = int(head[0])
    nt = int(head[1])

    pts = a[1:nv+1,:]
    pts = np.array(pts[:,0:2])
    tri = np.array(a[nv+1:nv+nt+1,:],dtype=int)

    tri = tri-1
    
    edges = np.array(a[nv+nt+1:,0:2],dtype=int)
    edges = edges-1
    
    n = np.loadtxt("ncells.dat",dtype=int)
    u = np.loadtxt("Part2D.data")
    
    nl = len(u)//n
    
    print(len(u))
    print(nl)
    
    U = np.zeros((nl,n))
    for i in range(0,n):
        U[:,i] = u[i*nl:(i+1)*nl]
    v = np.max(U,axis=1)

    plt.figure()
    plt.tripcolor(pts[:,0],pts[:,1],tri,v,shading='gouraud',
                  cmap='jet')
    plt.axis("off")
    plt.axis("scaled")
    plt.savefig("MMPartExample.pdf",bbox_inches='tight',pad_inches = -0.1)
    plt.show()
    
PlottingPart2D()   

def CostPlot2D():
    a = np.loadtxt("cost.dat")
    print(a)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(a)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cost")
    plt.savefig("Cost2D.pdf",bbox_inches='tight')
CostPlot2D()
