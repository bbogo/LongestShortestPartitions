import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
from shapely import geometry
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, MultiPoint, Polygon, Point
from shapely.geometry.polygon import LinearRing
from shapely.ops import polygonize
import scipy as scipy

data = np.loadtxt('coords.dat')
areaC = np.loadtxt('areas.dat')
#print(data.shape)

#plt.figure()
#plt.plot(data[:,0],data[:,1],'x')
#plt.axis("scaled")
#plt.show()

oP = Polygon(data)#.simplify(0.001,preserve_topology=False)

def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        v = [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
        random_point = Point(v)
        if (random_point.within(poly)):
            points.append(v)

    return np.array(points)

def extract_poly(vor,fact=1,debug=0):
    pts = vor.points  # get the points, each one has a region
    center = MultiPoint(vor.points).convex_hull.centroid
    center = np.array([center.x,center.y])
    vert = vor.vertices
    #print(vert)
    ridge_points = vor.ridge_points
    ridge_vertices = vor.ridge_vertices
    #print(ridge_points)
    #print(ridge_vertices)
    inf_ridges = np.zeros(len(ridge_vertices))
    for i in range(0,len(ridge_vertices)):
        inf_ridges[i]=sum(np.array(ridge_vertices[i])==-1)
    ListPoly = []
    
    boundary = np.zeros(len(pts))
    
    # build ridges first, then construct polygon... simpler
    #plt.figure()
    ridge_pts = []
    for i in range(0,len(ridge_points)):
        ridge = ridge_points[i]
        vridge = ridge_vertices[i]
        #print(ridge_vertices[i])
        inf_ridge = (-1 in ridge_vertices[i])
        if inf_ridge:
            for j in ridge_vertices[i]:
                if j!=-1:
                    pt2 = vert[j,:]
            i1 = ridge[0]
            i2 = ridge[1]
            # ridge 1   points i and q[0]
            v = pts[i1]-pts[i2]
            n = np.array([-v[1],v[0]])     # normal vector
            n = n/np.sqrt(np.sum(n**2))
            mid = (pts[i1,:]+pts[i2,:])/2
            cvec = mid-center
            if(np.dot(cvec,n)<0):
                n = -n
            pt1 = mid+fact*n
            #print(ridge)
            #plt.plot(pt1[0],pt1[1],'rx')
            #plt.plot(pt2[0],pt2[1],'bx')
            ridge_pts.append(np.array([pt2,pt1]))
        else:
            pt1 = vert[vridge[0],:]
            pt2 = vert[vridge[1],:]
            ridge_pts.append(np.array([pt1,pt2]))

            #print(pt1)
            #plt.plot(pt1[0],pt1[1],'gx')
            #plt.plot(pt2[0],pt2[1],'gx')
    #plt.show()       
    #print(ridge_pts)
    
    for i in range(0,len(pts)):
        reg = vor.regions[vor.point_region[i]]
        if -1 not in reg:
            ListPoly.append(vert[reg])
        else:
            #print("Infinite region:")
            #print(reg)
            # find ridges near the point p
            boundary[i]=1
            pt = pts[i]
            tS = [sum(a==i) for a in ridge_points]  # find point indexes for ridges near p
            tS = tS*inf_ridges                      # hold only infinite ridges
            qq = np.where(tS==1)                     # find ridge points indexes
            qq = qq[0]
            #print("Ridge points")
            q = []
            for j in qq:
                for k in range(0,2):
                    if ridge_points[j][k]!=i:
                        q.append(ridge_points[j][k])
                        

            # ridge 1   points i and q[0]
            
            act_pts = ridge_pts[qq[0]]
            
            pt1 = act_pts[1,:]
            
            
            
            # ridge 2   points i and q[1]
            act_pts = ridge_pts[qq[1]]
            
            pt2 = act_pts[1,:]
            
            poly = np.zeros((len(reg)+1,2))
            pos = 0
            for j in range(0,len(reg)):
                if reg[j]>=0:
                    poly[pos] = vert[reg[j]]
                    pos = pos+1
                else:
                    poly[pos] = pt2
                    pos = pos+1
                    poly[pos] = pt1
                    pos = pos+1
            if debug==1:
                plt.figure()
                print(poly)
                plt.plot(pts[:,0],pts[:,1],'rx')
                plt.plot(vert[:,0],vert[:,1],'b.')
                tpoly = np.vstack((poly,poly[0,:]))
                plt.plot(tpoly[:,0],tpoly[:,1])
                plt.show()
            
            c = poly.mean(axis=0)
            angles = np.arctan2(poly[:,1] - c[1], poly[:,0] - c[0])
            #print(np.argsort(angles))
            newpoly = poly[np.argsort(angles),:]    
            ListPoly.append(poly)            
    return ListPoly, boundary, ridge_pts



def PolyVoronoi(coords,poly,fact=10,plotting=0,debug=0):
    vor = Voronoi(coords)                 # voronoi diag
    
    # search for outer points and set them to -1??
    
    
    ListPoly,bnd,ridge_pts = extract_poly(vor,fact=fact,debug=debug)  # extract polygons
    oP = Polygon(poly)
    
    bnd = np.zeros(len(coords))
    poly_shapes = []
    pos = 0
    for p in  ListPoly:
        zz = ConvexHull(p);
        qz = zz.points[zz.vertices]
        qp = Polygon(qz)
        qpint =qp.intersection(oP)
        #print(qp)
        poly_shapes.append(qpint)
        #print(pos," ",qp.area," ",qpint.area)
        if qp.area>qpint.area:
            bnd[pos]=1
        pos = pos+1
        
    if plotting==1:
        plt.figure()
    Areas = np.zeros(len(poly_shapes))
    for i in range(0,len(poly_shapes)):
        if plotting==1:
            plt.plot(coords[i,0],coords[i,1],'rx')
        if poly_shapes[i].is_empty == 0:
            if plotting==1:
                x,y = poly_shapes[i].exterior.xy
                plt.plot(x,y)
            Areas[i] = poly_shapes[i].area;
    if plotting==1:
        plt.axis("scaled")
        plt.show()

    return poly_shapes, bnd, Areas, vor, ridge_pts  


def AreaFuncGrad(coords,poly):
    coords = np.reshape(coords,(-1,2))
    ps, bnd, areas, vor, ridge_vert = PolyVoronoi(coords,poly,debug=0,fact=10)
    
    area = np.sum(areas)
    grad = np.zeros((2*len(coords),len(coords)))
    
    ngrad = np.zeros((2*len(coords),len(coords)))
    tgrad = np.zeros((2*len(coords),len(coords)))

    
    #plt.figure()
    e1 = np.array([1,0])
    e2 = np.array([0,1])
    # for any ridge get the corresponding side and projection
    for i in range(0,len(vor.ridge_points)):
        vridge = ridge_vert[i]
        ridge = vor.ridge_points[i]
        
        i1 = ridge[0]
        i2 = ridge[1]
        point1 = coords[i1,:]
        point2 = coords[i2,:]
        seg = point2-point1
        lseg = np.linalg.norm(seg)
        normal = (seg)/np.linalg.norm(seg) # normal pointing from cell 1 towards cell 2
        
        #print(ridge)
        pridge = LineString([vridge[0,:],vridge[1,:]])
        common = pridge.intersection(poly)       # common edge
        #print(common)
        #print(common)
        if common.is_empty==0:
            ccoords = np.array(common.coords)
            #print(ccoords)
            #plt.plot(ccoords[:,0],ccoords[:,1])
            mid = (point1+point2)/2
        
            vseg = ccoords[1,:]-ccoords[0,:]
        
            orvec = np.dot(np.append(seg,0),np.append(vseg,0))
            orient = 1#np.sign(orvec)
            #print("Orient=",orient)
        
            len1 = np.linalg.norm(mid-ccoords[0,:])
            len2 = np.linalg.norm(mid-ccoords[1,:])
            length = np.linalg.norm(vseg)
            #print("Length=",length)
            # normal part 
            nv = np.dot(e1,normal)/2*length
            ngrad[2*i1,i1] = ngrad[2*i1,i1]+nv     # cell i1
            ngrad[2*i2,i1] = ngrad[2*i2,i1]+nv    

            ngrad[2*i1,i2] = ngrad[2*i1,i2]-nv     # cell i2
            ngrad[2*i2,i2] = ngrad[2*i2,i2]-nv            

            nv = np.dot(e2,normal)/2*length
            ngrad[2*i1+1,i1] = ngrad[2*i1+1,i1]+nv     # cell i1
            ngrad[2*i2+1,i1] = ngrad[2*i2+1,i1]+nv    

            ngrad[2*i1+1,i2] = ngrad[2*i1+1,i2]-nv     # cell i2
            ngrad[2*i2+1,i2] = ngrad[2*i2+1,i2]-nv    

            # tangential part

            tv = orient*np.dot(e1,vseg)/2/lseg/length*(len2**2-len1**2)
            tgrad[2*i1,i1] = tgrad[2*i1,i1]+tv     # cell i1
            tgrad[2*i2,i1] = tgrad[2*i2,i1]-tv      

            tgrad[2*i1,i2] = tgrad[2*i1,i2]-tv     # cell i2
            tgrad[2*i2,i2] = tgrad[2*i2,i2]+tv      


            tv = orient*np.dot(e2,vseg)/2/lseg/length*(len2**2-len1**2)
            tgrad[2*i1+1,i1] = tgrad[2*i1+1,i1]+tv     # cell i1
            tgrad[2*i2+1,i1] = tgrad[2*i2+1,i1]-tv    

            tgrad[2*i1+1,i2] = tgrad[2*i1+1,i2]-tv     # cell i2
            tgrad[2*i2+1,i2] = tgrad[2*i2+1,i2]+tv    

            #print(ccoords)
    #plt.show()
    grad = tgrad+ngrad
    return areas, grad


from scipy.optimize import minimize



def EqualAreasVoronoiV2(init,poly,nit=10,debug=0,eps=1e-2,fact=100,nitLloyd=0):
    totarea = poly.area;
    meanarea = totarea/len(init) # target area for one cell

    meanareas = totarea*areaC

    coords = init
    for it in range(0,nitLloyd):
        ps, bnd, areas, vor, ridge_points = PolyVoronoi(coords,poly,debug=debug,fact=fact)
        for i in range(0,len(coords)):
            coords[i] = ps[i].centroid.coords
    # use scipy minimize
        
    def ObjFunc(x,p=1.1):
        x = np.reshape(x,(-1,2))
        areas, gradA = AreaFuncGrad(x,poly)
        gradt = np.zeros(x.shape)
        if np.min(areas<1e-6):
            return 10**16,np.random(np.prod(np.shape(x)))

        #part = np.sum(np.abs(areas-meanarea)**p)**(1/p-1)
        for i in range(0,len(x)):
            gradt = gradt+2*(areas[i]-meanareas[i])*np.reshape(gradA[:,i],(-1,2))
        return (np.sum((areas-meanareas)**2)), np.reshape(gradt,(-1,1))
    
    def GradFunc(x):
        x = np.reshape(x,(-1,2))
        areas, gradA = AreaFuncGrad(x,poly)
        gradt = np.zeros(x.shape)
        for i in range(0,len(x)):
            gradt = gradt+2*(areas[i]-meanareas[i])*np.reshape(gradA[:,i],(-1,2))
        return np.reshape(gradt,(-1,1))
    
    init = np.reshape(coords,(-1,1))
    res = minimize(ObjFunc,init,method='L-BFGS-B',jac=True)
    
    coords = np.reshape(res.x,(-1,2))
    #PolyVoronoi(coords,poly,plotting=1,fact=fact)
    #print("Final Areas:")
    a, g = AreaFuncGrad(coords,poly)
    #print(a)
    #print("Final obj: ",val)
    return coords


def AreaPerimVoronoi(coords,poly):
    coords = np.reshape(coords,(-1,2))
    ps, bnd, areas, vor, ridge_pts = PolyVoronoi(coords,poly,debug=0,fact=10)
    perims = np.zeros(len(ps))
    for i in range(0,len(ps)):
        if ps[i].is_empty == 0:
            x,y = ps[i].exterior.xy
            perims[i] = np.sum(np.sqrt((x-np.roll(x,1))**2+(y-np.roll(y,1))**2))
            
    return areas, perims


def InitProposal(poly,ncells,ntries=10,nitLloyd=20):
    poly = Polygon(poly)
    best = 10**16
    for i in range(0,ntries):
        coords = random_points_within(poly,ncells) 
        totarea = Polygon(poly).area
        meanarea = totarea/ncells
        res = EqualAreasVoronoiV2(coords,poly,nit=100,eps=1e-1,nitLloyd=nitLloyd)
        epst = 0
        #print(res)
        areas, perims = AreaPerimVoronoi(res,poly)
        val = np.sum(perims)
        #print("current value: ",val)

        if val<best:
            Candidate = res
            best = val
    print("Best value: ",best)
    return Candidate

ncells = np.loadtxt('ncells.dat')
print(ncells)

Candidate = InitProposal(oP,ncells,ntries=20,nitLloyd=0) 
#PolyVoronoi(Candidate,oP,plotting=1,fact=100)

np.savetxt('VoronoiPoints.dat', Candidate)



