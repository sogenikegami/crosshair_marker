import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

#input: hamiltonian, vertexdata

def Projector(H,fermi_energy):     # define and calculate the Projection P
    eigval,eigvec = np.linalg.eig(H)
    N = len(eigval)
    P = np.zeros((N,N))*0j
    for i in range(N):
        if eigval[i].real <= fermi_energy:
            eigenvector = np.zeros((N,1))*0j
            for j in range(N):
                eigenvector[j][0] += eigvec[j][i] 
            P += eigenvector * eigenvector.conj().T
    return P

def thetax(H,vertexdata,Rx):    # define and calculate the step and Projection theta_x , theta_y
    N = len(H[0])
    inner_deg_free = int(N/len(vertexdata))
    theta_x = np.zeros((N,N))*0j
    for j in range(inner_deg_free):
        for i in range(len(vertexdata)):
            if vertexdata[i]["pos"][0] > Rx:
                theta_x[i+j*len(vertexdata)][i+j*len(vertexdata)] += 1
    return theta_x

def thetay(H,vertexdata,Ry):    # define and calculate the step and Projection theta_x , theta_y
    N = len(H[0])
    inner_deg_free = int(N/len(vertexdata))
    theta_y = np.zeros((N,N))*0j
    for j in range(inner_deg_free):
        for i in range(len(vertexdata)):
            if vertexdata[i]["pos"][1] > Ry:
                theta_y[i+j*len(vertexdata)][i+j*len(vertexdata)] += 1
    return theta_y

def deltar(H,vertexdata,rx,ry):    #take the local trace and calculate the crosshair marker
    N = len(H[0])
    delta_r = np.zeros((N,N))*0j
    inner_deg_free = int(N/len(vertexdata))
    for j in range(inner_deg_free):
        for i in range(len(vertexdata)):
            pos = vertexdata[i]["pos"]
            if (pos[0]-float(rx))*(pos[0]-float(rx)) + (pos[1]-float(ry))*(pos[1]-float(ry)) < 10**(-5):
                delta_r[i+j*len(vertexdata)][i+j*len(vertexdata)] += 1
    return delta_r

def crosshair(H,vertexdata,fermi_energy,Rx,Ry,rx,ry):
    P = Projector(H,fermi_energy)
    delta_r = deltar(H,vertexdata,rx,ry)
    theta_x = thetax(H,vertexdata,Rx)
    theta_y = thetay(H,vertexdata,Ry)
    M = np.dot(np.dot(np.dot(np.dot(np.dot(delta_r,P),theta_x),P),theta_y),P)
    crosshair_marker =  4*math.pi*np.trace(M).imag
    return crosshair_marker

def plot(H,vertexdata,fermi_energy,Rx,Ry,filepath,imgname):
    N = len(vertexdata)
    X = []
    Y = []
    marker = []
    fig = plt.figure(figsize=(8.66+2,10))
    ax = fig.add_subplot(1,1,1)
    for i in range(N):
        x = vertexdata[i]["pos"][0]
        y = vertexdata[i]["pos"][1]
        X.append(x)
        Y.append(y)
        marker.append(crosshair(H,vertexdata,fermi_energy,Rx,Ry,x,y))
        neighbor = vertexdata[i]["neighbor"]
        for j in range(len(neighbor)):
            neinum = vertexdata[i]["neighbor"][j]
            neipos = vertexdata[neinum]["pos"]
            nowpos = vertexdata[i]["pos"]
            plt.plot([nowpos[0],neipos[0]],[nowpos[1],neipos[1]],linewidth = 1,color="black")
    xmin = X[0]
    xmax = X[-1]
    ymin = Y[0]
    ymax = Y[-1]
    markermax = max(np.abs(marker))
    plt.hlines([Ry],xmin,xmax,linestyle="dashed",color='red')
    plt.vlines([Rx],ymin,ymax,linestyle="dashed",color='red')
    mappable = ax.scatter(X,Y,c=marker,cmap='bwr',s=800*np.abs(marker)/markermax,alpha=1,linewidth=0.5,edgecolors='black')
    #mappable = ax.scatter(X,Y,c=marker,cmap='bwr',s=1000,alpha=0.5,linewidth=0.5,edgecolors='black')
    fig.colorbar(mappable,ax=ax)
    fig.show()
    plt.savefig(filepath + imgname)

def local_marker(H,vertexdata,fermi_energy,filepath,imgname):
    N = len(vertexdata)
    X = []
    Y = []
    local_map = np.zeros(N)
    fig = plt.figure(figsize=(8.66+2,10))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(vertexdata)):
        pos = vertexdata[i]["pos"]
        Rx = pos[0]
        Ry = pos[1]
        X.append(Rx)
        Y.append(Ry)
        for j in range(len(vertexdata[i]["neighbor"])):
            neinum = vertexdata[i]["neighbor"][j]
            neipos = vertexdata[neinum]["pos"]
            plt.plot([pos[0],neipos[0]],[pos[1],neipos[1]],linewidth = 1,color="black")
        for j in range(len(vertexdata)):
            rpos = vertexdata[j]["pos"]
            local_map[j] += crosshair(H,vertexdata,fermi_energy,Rx,Ry,rpos[0],rpos[1])
    markermax = max(np.abs(local_map))
    mappable = ax.scatter(X,Y,c=local_map,cmap='bwr',s=800*np.abs(local_map)/markermax,alpha=1,linewidth=0.5,edgecolors='black')
    #mappable = ax.scatter(X,Y,c=marker,cmap='bwr',s=1000,alpha=0.5,linewidth=0.5,edgecolors='black')
    fig.colorbar(mappable,ax=ax)
    fig.show()
    plt.savefig(filepath + imgname)


def main():
    return 0

if __name__ == "__main__":
    main()
