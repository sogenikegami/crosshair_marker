import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.colors as mcolors
import seaborn as sns
import crosshair_marker01
import sys
sys.path.append('..')
from local_chern_marker import local_chern_marker01

fermi_energy = 0
Nx = 30 # systemsize
Ny = 30 #systemsize

Rx = 15 #crosshair pos
Ry = 15 #crosshair pos
r = 10 #Chern number=1 domain radius

filepath = "/Users/sogenikegami/Documents/UT4S/non-crystal/crosshair/image/"


def cartesian(Nx,Ny,a):
    vertexdata = []
    dn = [1,-1,Nx,-Nx]
    for j in range(Ny):
        for i in range(Nx):
            pos = np.array([a*i,a*j])
            neighbor = []
            for k in range(4):
                if 0 <= j*Nx+i+dn[k] and j*Nx+i+dn[k] < Nx*Ny:
                    if (i==0 and dn[k]==-1) or (i==Nx-1 and dn[k]==1):
                        continue
                    else:
                        neighbor.append(j*Nx+i+dn[k])
            vertexdata.append({"pos":pos,"neighbor":neighbor}) 
    return vertexdata

def plot(vertexdata):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    N = len(vertexdata)
    for i in range(N):
        for j in range(len(vertexdata[i]["neighbor"])):
            neinum = vertexdata[i]["neighbor"][j]
            neipos = vertexdata[neinum]["pos"]
            nowpos = vertexdata[i]["pos"]
            plt.plot([nowpos[0],neipos[0]],[nowpos[1],neipos[1]])
    fig.show()
    plt.savefig("/Users/sogenikegami/Documents/UT4S/non-crystal/crosshair/image/cartesian01")
    
def hplusx(Nx,Ny):
    N = Nx*Ny
    H = np.zeros((N*2,N*2))*0j
    for i in range(N):
        if i % Nx != Nx-1:
            pos = np.zeros((N,1))*0j
            pos[i][0] += 1
            r_plus_x = np.zeros((N,1))*0j
            r_plus_x[i+1][0] += 1
            position_matrix = r_plus_x * pos.conj().T
            a = np.hstack([position_matrix*0.5, position_matrix*0.5j])
            b = np.hstack([position_matrix*0.5j, position_matrix*(-0.5)])
            hamiltonian_component = np.vstack([a,b])
            H += hamiltonian_component
            H += hamiltonian_component.conj().T
    return H

def hplusy(Nx,Ny):
    N = Nx*Ny
    H = np.zeros((N*2,N*2))*0j
    for i in range(N-Nx):
        pos = np.zeros((N,1))*0j
        pos[i][0] += 1
        r_plus_y = np.zeros((N,1))*0j
        r_plus_y[i+Nx][0] += 1
        position_matrix = r_plus_y * pos.conj().T
        a = np.hstack([position_matrix*0.5, position_matrix*0.5])
        b = np.hstack([position_matrix*(-0.5), position_matrix*(-0.5)])
        hamiltonian_component = np.vstack([a,b])
        H += hamiltonian_component
        H += hamiltonian_component.conj().T
    return H

def hsite(Nx,Ny,r):
    N = Nx*Ny
    H = np.zeros((N*2,N*2))*0j
    for i in range(N):
        x = i % Nx
        y = (i-x)/Nx
        if (x-Nx/2)*(x-Nx/2) + (y-Ny/2)*(y-Ny/2) <= r*r:
            u = 1.6
        else:
            u = 2.6
        position_matrix = np.zeros((N,N))*0j
        position_matrix[i][i] += 1
        a = np.hstack([position_matrix*u, position_matrix*0])
        b = np.hstack([position_matrix*0, position_matrix*(-1.0)*u])
        hamiltonian_component = np.vstack([a,b])
        H += hamiltonian_component
    return H

def Hamiltonian(Nx,Ny,r):
    H = hplusx(Nx,Ny) + hplusy(Nx,Ny) + hsite(Nx,Ny,r)
    return H

def eigenergy(H,imgname):
    eigval,eigvec = np.linalg.eig(H)
    eigenergy = []
    for i in range(len(eigval)):
        eigenergy.append(eigval[i].real)
    eigenergy.sort()
    num = list(range(0,len(eigval)))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(num,eigenergy,s=10)
    fig.show()
    plt.savefig(filepath +  imgname)

def plotC(vertexdata,local_c,filepath,imgname,Nx,Ny,r):
    N = len(vertexdata)
    X = []
    Y = []
    theta = np.linspace(0,2*math.pi,200)
    circlex = Nx/2 + r * np.cos(theta)
    circley = Ny/2 + r * np.sin(theta)
    fig = plt.figure(figsize=(8.66+2,10))
    ax = fig.add_subplot(1,1,1)
    for i in range(N):
        X.append(vertexdata[i]["pos"][0])
        Y.append(vertexdata[i]["pos"][1])
        for j in range(len(vertexdata[i]["neighbor"])):
            neinum = vertexdata[i]["neighbor"][j]
            neipos = vertexdata[neinum]["pos"]
            nowpos = vertexdata[i]["pos"]
            plt.plot([nowpos[0],neipos[0]],[nowpos[1],neipos[1]],color="black",linewidth = 1)
    markermax = max(np.abs(local_c))
    mappable = ax.scatter(X,Y,c=local_c,cmap='bwr',s=800*np.abs(local_c)/markermax,alpha=1,linewidth=0.5,edgecolors='black')
    fig.colorbar(mappable,ax=ax)
    ax.plot(circlex,circley,color = "green",linestyle = "dashed")
    fig.show()
    plt.savefig(filepath + imgname)
    return 0


def main():
    vertexdata = cartesian(Nx,Ny,1)
    H = Hamiltonian(Nx,Ny,r)
    imgname = "QWZ-model_localC_from_ch01"
    imgname2 = "QWZ_model_crosshair_marker04" 
    #crosshair_marker01.plot(H,vertexdata,fermi_energy,Rx,Ry,filepath,imgname2)
    #crosshair_marker01.local_marker(H,vertexdata,fermi_energy,filepath,imgname)
    #eigenergy(H,imgname2)

    local_c = local_chern_marker01.local_chern_marker(H,vertexdata,fermi_energy)
    imgname3 = "QWZ-localC02"
    plotC(vertexdata,local_c,filepath,imgname3,Nx,Ny,r)
    #local_chern_marker01.plot(H,vertexdata,fermi_energy,filepath,imgname3)
    return 0


if __name__ == "__main__":
    main()