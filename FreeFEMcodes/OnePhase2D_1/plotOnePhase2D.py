import numpy as np
import matplotlib.pyplot as plt

# Plot from msh file
def PlottingOnePhase2D():

    # load information from msh file ass array
    # process to get all information regarding
    # points, triangles and edges
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
    
    # load the density information
    u = np.loadtxt("OnePhase2D.data")

    # plot the figure
    plt.figure()
    plt.tripcolor(pts[:,0],pts[:,1],tri,u,shading='gouraud',cmap='jet')
    plt.axis("off")
    plt.axis("scaled")
    # save the result: change the extension to png and add "dpi=150" to get an image instead of a pdf file
    plt.savefig("MMOnePhaseExample.pdf",bbox_inches='tight',pad_inches = -0.1)
    plt.show()
    
PlottingOnePhase2D()  

# Plot the cost information
def CostPlot2D():
    # load information and plot it
    a = np.loadtxt("cost.data")
    print(a)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(a)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cost")
    # Save as pdf file; change extension and add dpi information to get an image instead
    plt.savefig("Cost2D.pdf",bbox_inches='tight')
CostPlot2D()
