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
import nlopt

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



def PolyVoronoi(coords,poly,fact=10,plotting=0,debug=0,save_string=None):
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
            plt.plot(coords[i,0],coords[i,1],color='black', marker='.')
        if poly_shapes[i].is_empty == 0:
            if plotting==1:
                x,y = poly_shapes[i].exterior.xy
                plt.fill(x,y,edgecolor='black')
            Areas[i] = poly_shapes[i].area;
    if plotting==1:
        plt.axis("scaled")
        plt.axis('off')
        if save_string!=None:
            plt.savefig(save_string+'.pdf',bbox_inches='tight',pad_inches = -0.1)
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

def CircumcenterDer(coords,DirA):
    
    A = coords[0,:]
    B = coords[1,:]
    C = coords[2,:]
    
    D = 2*(A[0]*(B[1]-C[1])+B[0]*(C[1]-A[1])+C[0]*(A[1]-B[1]))
    
    Nx = (np.sum(A**2)*(B[1]-C[1])+np.sum(B**2)*(C[1]-A[1])+np.sum(C**2)*(A[1]-B[1]))
    Ny = -(np.sum(A**2)*(B[0]-C[0])+np.sum(B**2)*(C[0]-A[0])+np.sum(C**2)*(A[0]-B[0]))
    
    Ox = 1/D*Nx
    Oy = 1/D*Ny
    
    
    Dx = 2*(B[1]-C[1])
    Dy = 2*(C[0]-B[0])
    
    Oxx = -Dx/D**2*Nx+1/D*2*A[0]*(B[1]-C[1])
    Oxy = -Dy/D**2*Nx+1/D*(2*A[1]*(B[1]-C[1])-np.sum(B**2)+np.sum(C**2))
    
    Derx = np.dot([Oxx,Oxy],DirA)
    
    Oyx = -Dx/D**2*Ny+1/D*(2*A[0]*(C[0]-B[0])+np.sum(B**2)-np.sum(C**2))
    Oyy = -Dy/D**2*Ny+1/D*2*A[1]*(C[0]-B[0])
    
    Dery = np.dot([Oyx,Oyy],DirA)
    
    Center = np.array([Ox,Oy])
    return Center, np.array([Derx,Dery])

def PerimFuncGrad(coords,poly,plotting=0):
    from numpy.linalg import norm
    coords = np.reshape(coords,(-1,2))
    ps, bnd, areas, vor, ridge_vert = PolyVoronoi(coords,poly,debug=0,fact=100,plotting=plotting)
    
    #print(ridge_vert)
    
    perims = np.zeros(len(ps))
    for i in range(0,len(ps)):
        if ps[i].is_empty == 0:
            x,y = ps[i].exterior.xy
            perims[i] = np.sum(np.sqrt((x-np.roll(x,1))**2+(y-np.roll(y,1))**2))
    
    area = np.sum(areas)
    grad = np.zeros((2*len(coords),len(coords)))
    
    ngrad = np.zeros((2*len(coords),len(coords)))
    tgrad = np.zeros((2*len(coords),len(coords)))
    
    e1 = np.array([1,0])
    e2 = np.array([0,1])
    
    #print("Perims: ",perims)
    
    vertices = vor.vertices
    ridge_vertices = vor.ridge_vertices
    ridge_pts = vor.ridge_points
    #print(vertices)
    Poly = Polygon(poly)
    
    #print(ridge_vertices)
    #print(ridge_pts)
    # construct list of ridges with given vertex
    
    
    for i in range(0,len(vertices)):
        # test if current vertex is inside the polygon!
        vert = vertices[i]   # store current vertex
        if Poly.contains(Point(vertices[i,:])):
            #print("Inside")
            vL = []
            # find index of ridges containing vertex i
            # need indices to be able to find the points easily (they have the same order)
            for ind in range(0,len(ridge_vertices)):
                if i in ridge_vertices[ind]:
                    vL.append(ind)
            #print("Ridges containing vertex[",i,"]: ",vL)
            
            # point list
            pL = []
            closePts = np.unique(ridge_pts[vL,:])
            for j in range(0,len(closePts)):
                p0 = closePts[j]
                restr_ridge_pts = ridge_pts[vL,:]
                #print(p0," and ",restr_ridge_pts)
                tS = np.sum(restr_ridge_pts==p0,axis=1)
                qq = np.nonzero(tS==1)[0]
                #print("tS=",qq)
                tri = [p0]
                for k in qq:
                    if restr_ridge_pts[k,0] != p0:
                        tri.append(restr_ridge_pts[k,0])
                    else:
                        tri.append(restr_ridge_pts[k,1])
                # now tri contains the three points wiht vertex p0 on first position!
                # compute sensitivity of voronoi vertex (circumcenter) w.r.t. perturbation in p0
                
                tri_coords = coords[tri,:]
                unused, Der1 = CircumcenterDer(tri_coords,e1)
                unused, Der2 = CircumcenterDer(tri_coords,e2)
                                
                # for each ridge in triangle
                # project Der on it and add the gradient contribution to the grad of two cells 
                
                # loop on ridges coming to current vertex
                #plt.figure()
                for k in range(0,len(vL)):
                    # need vector of the rdige ENDING in vertices[i]!
                    
                    if 1==1:
                    
                        vert = vertices[i] # current vertex
                        rVert = ridge_vert[vL[k]] # ridge vertices (possibly longer)
                        rPoint = ridge_pts[vL[k]] # to see which cells are affected

                        #print("Ridge vertices vL[k] ",ridge_vertices[vL[k]])
                        #print(rVert)

                        act_ridge = ridge_vertices[vL[k]]
                        if -1 not in act_ridge:
                            if act_ridge[0]!=i: #if first vertex of ridge is not i
                                other = rVert[0,:]
                            else:
                                other = rVert[1,:]
                        else:
                            other = rVert[1,:]
                        ridgeVect = vert-other
                        ridgeVect = ridgeVect/norm(ridgeVect)

                        # compute gradient for current ridge
                        gx = np.dot(ridgeVect,Der1)
                        gy = np.dot(ridgeVect,Der2)

                        # add contribution
                        grad[2*p0,rPoint[0]]=grad[2*p0,rPoint[0]]+gx
                        grad[2*p0,rPoint[1]]=grad[2*p0,rPoint[1]]+gx

                        grad[2*p0+1,rPoint[0]]=grad[2*p0+1,rPoint[0]]+gy
                        grad[2*p0+1,rPoint[1]]=grad[2*p0+1,rPoint[1]]+gy

                        #rPlot = np.vstack((vert,other))
                        #plt.plot(rPlot[:,0],rPlot[:,1])
                
                #plt.show()
        
            #print("Outside: nothing to do")
        
        # loop on ridges and find if they intersect the outer polygon
    
    
    
    circpoly = np.vstack((poly,poly[0,:]))
    lPoly = LineString(circpoly)
    
    #print(lPoly)
    
    for i in range(0,len(ridge_vertices)):
        #print(ridge_vertices[i])
        #print(ridge_vert[i])
        ridge_line = LineString((a[0],a[1]) for a in ridge_vert[i])
        geom = ridge_line.intersection(lPoly)
        rPoint = ridge_pts[i]
        if geom.is_empty==0:
            #plt.plot(ridge_vert[i][:,0],ridge_vert[i][:,1])
            #print(geom)
            
            # find segment
            for j in range(0,len(circpoly)-1):
                A = circpoly[j,:]
                B = circpoly[j+1,:]
                actLine = LineString([(A[0],A[1]),(B[0],B[1])])
                if actLine.contains(geom):
                    break
            
            
            act = ridge_pts[i]
            i1 = act[0]
            i2 = act[1]
            
            p1 = coords[i1,:]
            p2 = coords[i2,:]
            
            orient = np.sign(np.dot(p1-p2,A-B))
            lengthAB = norm(A-B)
            ABvec = orient*(A-B)/lengthAB
            
            # moving p1, reflect p2 w.r.t. AB
            proj = B+(A-B)*np.dot(A-B,p2-B)/lengthAB**2
            refl = 2*proj-p2
            
            tri_coords = np.vstack((p1,p2,refl))
                    
            # perturb vertex p1
            unused, Der1 = CircumcenterDer(tri_coords,e1)
            unused, Der2 = CircumcenterDer(tri_coords,e2)
            
            # compute contributions
            ridgeVect = ridge_vert[i][1,:]-ridge_vert[i][0,:]
            ridgeVect = ridgeVect/norm(ridgeVect)
            ridgeVect = np.sign(np.dot(ridgeVect,refl-p2))*ridgeVect
            
            # add contribution for ridge
            
            gx = np.dot(ridgeVect,Der1)
            gy = np.dot(ridgeVect,Der2)
            
            
            
            ABvec = -ABvec
            # add tangential contribution
            tx = np.dot(ABvec,Der1)
            ty = np.dot(ABvec,Der2)
            
            grad[2*i1,rPoint[0]]=grad[2*i1,rPoint[0]]+gx+tx
            grad[2*i1,rPoint[1]]=grad[2*i1,rPoint[1]]+gx-tx

            grad[2*i1+1,rPoint[0]]=grad[2*i1+1,rPoint[0]]+gy+ty
            grad[2*i1+1,rPoint[1]]=grad[2*i1+1,rPoint[1]]+gy-ty
            
            
            # reflect p1
            proj = B+(A-B)*np.dot(A-B,p1-B)/lengthAB**2
            refl = 2*proj-p1
            tri_coords = np.vstack((p2,p1,refl))

            # perturb vertex p2
            unused, Der1 = CircumcenterDer(tri_coords,e1)
            unused, Der2 = CircumcenterDer(tri_coords,e2)
            
            gx = np.dot(ridgeVect,Der1)
            gy = np.dot(ridgeVect,Der2)
            
            
            
            # add tangential contribution
            tx = np.dot(ABvec,Der1)
            ty = np.dot(ABvec,Der2)
            
            grad[2*i2,rPoint[0]]=grad[2*i2,rPoint[0]]+gx+tx
            grad[2*i2,rPoint[1]]=grad[2*i2,rPoint[1]]+gx-tx

            grad[2*i2+1,rPoint[0]]=grad[2*i2+1,rPoint[0]]+gy+ty
            grad[2*i2+1,rPoint[1]]=grad[2*i2+1,rPoint[1]]+gy-ty
        
    return perims, grad

def EqualAreasVoronoiMinPerim(init,poly,fact=100,areaC=None,nitLloyd=10,save_string=None):
    totarea = poly.area;
    
    ncells = len(init)
    
    meanarea = totarea/len(init) # target area for one cell
    meanareas = np.ones(len(init))/len(init)*totarea
    
    if(areaC is None):
        meanareas = np.ones(len(init))/len(init)*totarea
    else: 
        meanareas = areaC*totarea
   
    coords = init
    
    for it in range(0,nitLloyd):
        ps, bnd, areas, vor, ridge_points = PolyVoronoi(coords,poly,fact=fact)
        for i in range(0,len(coords)):
            coords[i] = ps[i].centroid.coords
            
    xx,yy = poly.exterior.xy
    cc  = np.zeros((len(xx)-1,2))
    cc[:,0]=xx[0:-1]
    cc[:,1]=yy[0:-1]

                
    def myfunc(x, grad):
        vec = np.reshape(x,(-1,2))
        
        perims, gradP = PerimFuncGrad(vec,cc,plotting=0)
        gradt = np.zeros(vec.shape)
        val = np.sum(perims)
        
        for i in range(0,ncells):
            gradt = gradt+np.reshape(gradP[:,i],(-1,2))
        if grad.size > 0:
            grad[:] = gradt.flatten()
        if(np.min(perims)<-1):
            centroid = np.mean(np.reshape(x,(-1,2)),axis=0)
            centroids = np.zeros((len(x)/2,2))
            centroids[:,0] = centroid[0]
            centroids[:,1] = centroid[1]
            val = 10**3
            grad[:] = -0.1*(x-np.reshape(centroids,(-1,1)))
        return val
    def c(result, x, grad):
        vec = np.reshape(x,(-1,2))
        
        areas, gradA = AreaFuncGrad(vec,poly)
        result[:] = (areas-meanareas)
        if grad.size > 0:
            for i in range(0,len(init)):
                grad[i,:] = gradA[:,i]
    opt = nlopt.opt(nlopt.LD_MMA, 2*ncells)
    opt_interm = nlopt.opt(nlopt.LD_MMA, 2*ncells)
    opt_interm.set_xtol_rel(1e-8)
    opt_interm.set_ftol_rel(1e-8)
    opt_interm.set_maxeval(100)
    opt.set_local_optimizer(opt_interm)
    opt.set_maxeval(100)
    opt.set_min_objective(myfunc)
    tol = 1e-6*np.ones(len(init))
    opt.add_inequality_mconstraint(c, tol)
    opt.set_xtol_rel(1e-6)
    opt.set_ftol_rel(1e-6)
    initialization = init.flatten()
    #print(initialization)
    xres = opt.optimize(initialization)
    minf = opt.last_optimum_value()
    #print("optimum at ", xres)
    #print("minimum value = ", minf)
    #print("result code = ", opt.last_optimize_result())        
        
        
        
    
    coords = np.reshape(xres,(-1,2))
    #PolyVoronoi(coords,poly,plotting=1,fact=fact,save_string=save_string)
    #print("Final Areas:")
    #a, g = AreaFuncGrad(coords,poly)
    #print(a)
    
    return coords


def InitProposalPerim(poly,ncells,ntries=10,areaC=None,nitLloyd=20):
    poly = Polygon(poly)
    best = 10**16  
    it   = np.loadtxt('iter.dat')
    vals = []
    Candidates = []
    if (it==1):
        ntries=20
    if (it>=2):
	     #load the best three candidates

        Candidate1 = np.loadtxt('Candidate1.dat')
        try:
             Candidate1 = EqualAreasVoronoiMinPerim(Candidate1,oP,nitLloyd=0,areaC=areaC)
        except: 
             print("Nlopt failed!") 
             Candidate1 = EqualAreasVoronoiV2(Candidate1,poly)
        areas, perims = AreaPerimVoronoi(Candidate1,poly)
        val1 = np.sum(perims)
        if np.min(areas)<1e-3:
            # invalid partition
            print("Invalid partition!!")
            val1 = 10**16
        vals.append(val1)
        Candidates.append(Candidate1)
        Candidate2 = np.loadtxt('Candidate2.dat')
        try:
             Candidate2 = EqualAreasVoronoiMinPerim(Candidate2,oP,nitLloyd=0,areaC=areaC)
        except: 
             print("Nlopt failed!") 
             Candidate2 = EqualAreasVoronoiV2(Candidate2,poly)
        areas, perims = AreaPerimVoronoi(Candidate2,poly)
        val2 = np.sum(perims)
        if np.min(areas)<1e-3:
            # invalid partition
            print("Invalid partition!!")
            val2 = 10**16
        vals.append(val2)
        Candidates.append(Candidate2)
        Candidate3 = np.loadtxt('Candidate3.dat')
        try:
             Candidate3 = EqualAreasVoronoiMinPerim(Candidate3,oP,nitLloyd=0,areaC=areaC)
        except:
             print("Nlopt failed!") 
             Candidate3 = EqualAreasVoronoiV2(Candidate3,poly)    
        areas, perims = AreaPerimVoronoi(Candidate3,poly)
        val3 = np.sum(perims)
        if np.min(areas)<1e-3:
            # invalid partition
            print("Invalid partition!!")
            val3 = 10**16
        vals.append(val3)
        Candidates.append(Candidate3)
        ind = np.argsort(vals)
        vals[:] = [vals[j] for j in ind]
        Candidates[:] = [Candidates[j] for j in ind]
    print("Iter: ",it," | Number of tries: ",ntries)    
    for i in range(0,ntries):
        coords = random_points_within(poly,ncells) 
        totarea = Polygon(poly).area
        meanarea = totarea/ncells
        try:
            res = EqualAreasVoronoiMinPerim(coords,poly,nitLloyd=5,areaC=areaC)
        except:
            print("Nlopt failed!") 
            res = EqualAreasVoronoiV2(coords,poly,nitLloyd=5) 
        epst = 0
        areas, perims = AreaPerimVoronoi(res,poly)
        val = np.sum(perims)
        if np.min(areas)<1e-3:
            # invalid partition
            print("Invalid partition!!")
            val = 10**16
        vals.append(val)
        Candidates.append(res)
        ind = np.argsort(vals)
        vals[:] = [vals[j] for j in ind]
        Candidates[:] = [Candidates[j] for j in ind]
    Candidate = Candidates[0]
    best = vals[0]
    # save best three
    np.savetxt('Candidate1.dat',Candidates[0])
    np.savetxt('Candidate2.dat',Candidates[1])
    np.savetxt('Candidate3.dat',Candidates[2])
    #PolyVoronoi(Candidate,poly,plotting=1,fact=100)
    print("Best value: ",best)
    return Candidate

ncells = np.loadtxt('ncells.dat')


Candidate = InitProposalPerim(oP,ncells,ntries=10,nitLloyd=20,areaC=areaC) 
#PolyVoronoi(Candidate,oP,plotting=1,fact=100)

np.savetxt('VoronoiPoints.dat', Candidate)



