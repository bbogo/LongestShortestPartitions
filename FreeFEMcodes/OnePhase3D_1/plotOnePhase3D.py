import meshio
import numpy as np
import matplotlib.pyplot as plt

mesh = meshio.read("CurrentMesh3D.mesh")

mdet = mesh.cells
tetra = mdet[0].data
tri   = mdet[1].data

pts = mesh.points
x = pts[:,0]
y = pts[:,1]
z = pts[:,2]

v = np.loadtxt("OnePhase3D.data")

import mayavi

from mayavi import mlab
from tvtk.api import tvtk

fig = mlab.figure(bgcolor=(1,1,1),size=(400, 350))

tet_type = tvtk.Tetra().cell_type

ug = tvtk.UnstructuredGrid(points=pts)
ug.point_data.scalars = v
ug.point_data.scalars.name = "value"
ug.set_cells(tet_type, tetra)
ds = mlab.pipeline.add_dataset(ug)
iso = mlab.pipeline.iso_surface(ds,contours=[0.5,],figure=fig)

iso.actor.property.opacity = 0.7
iso.contour.number_of_contours = 1
#mlab.show()

l = mlab.triangular_mesh(x,y,z,tri,color=(0.3,0.3,0.3),
                        opacity=0.4,figure=fig)

#mlab.savefig("OnePhase3D_1.png",size=(400,400))
mlab.show()


def CostPlot3D():
    a = np.loadtxt("cost3D.data")
    print(a)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(a)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cost")
    plt.savefig("Cost3D.pdf",bbox_inches='tight')
CostPlot3D()



