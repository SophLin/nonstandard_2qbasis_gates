import numpy as np
import cmath

Q = np.asarray(
        [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]
    ) / np.sqrt(2)
Q_H = Q.conj().T

def Makhlin_invariants(U):
    """
    Returns the Makhlin invariants G1, G2 of operator U. G1 is complex and G2 is real.
    """
    M_B = Q_H@U@Q
    m = M_B.T@M_B
    tr_m = m.trace()
    det_U = np.linalg.det(U)
    G1 = tr_m**2/(16*det_U)
    G2 = (tr_m**2 - (m@m).trace())/(4*det_U)
    return G1, G2

# check if two unitaries are locally equiavalent
def locally_equivalent(U1,U2, tol = 1e-9):
    G1_U1, G2_U1 = Makhlin_invariants(U1)
    G1_U2, G2_U2 = Makhlin_invariants(U2)
    if abs(G1_U1.real - G1_U2.real) > tol:
        return False
    if abs(G1_U1.imag - G1_U2.imag) > tol:
        return False
    if abs(G2_U1 - G2_U2) > tol:
        return False
    return True

def logspec_to_U(logspec):
    """
    Returns a matrix U such that the logspec of U's Cartan double is the input
    """
    F = np.diag(np.exp([1j*cmath.pi*logspec[0],1j*cmath.pi*logspec[1]
                 ,1j*cmath.pi*logspec[2],1j*cmath.pi*logspec[3]]))
    U = Q@F@Q_H
    return U

def Cartan_double(U):
    M_B = Q_H@U@Q
    m = M_B.T@M_B
    return m

# Calculate logSpec of the Cartan double of U
def U_to_logspec(U):
    U /= np.linalg.det(U) ** (1 / 4)
    cartan_double = Cartan_double(U)
    evals = np.linalg.eigvals(cartan_double)
    #print("Eigenvals:",evals)
    logspec = np.log(evals).imag/np.pi/2
    #print("log then divide by 2*pi*i",logspec)
    logspec = np.sort(logspec)
    logspec = logspec[::-1]
    #print("Returns",logspec)
    return logspec

#Convert Weyl chamber coordinates to LogSpec
#can: Weyl chamber coordinates of a gate 1
#Returns: logspec, and a flag that indicates whether the LogSpec follows the convention
def weyl_to_logspec(can):
    logspec = -1*np.array([can[0]-can[1]+can[2], can[0]+can[1]-can[2],
                           -1*can[0]-can[1]-can[2], -1.*can[0]+can[1]+can[2]])/2
    logspec.sort()
    logspec = logspec[::-1]
    convention = True #whether the Logspec satisfies the convention
    if abs(np.sum(logspec)) > 1e-9:
        convention = False
    elif logspec[3] < logspec[0] - 1:
        convention = False
    return logspec, convention

#rotate logspec into an equivalent one
def rotate(logspec):
    re = np.array([logspec[2]+1/2,logspec[3]+1/2,logspec[0]-1/2,logspec[1]-1/2])
    return re

def process_fid(x,y):
    """Norm of the Hilbert-Schmidt inner-product"""
    return np.abs((x.conj() * y).sum()/len(y))

def entangling_power(w):
    c1 = w[0] * np.pi/2
    c2 = w[1] * np.pi/2
    c3 = w[2] * np.pi/2
    return 1/18 * (3 - (np.cos(4*c1)*np.cos(4*c2) + np.cos(4*c2)*np.cos(4*c3) + np.cos(4*c3)*np.cos(4*c1)))

def in_region1(w):#w needs to be an np array
    """Check if a point is in the tetrahedron between CZ, I, sqrt(iSWAP), sqrt(SWAP)^dag"""
    p1 = np.array([1/2,0,0])#CNOT/CZ
    p2 = np.array([3/4,1/4,0]) #sqrt(iSWAP)
    p3 = np.array([3/4,1/4,1/4]) #sqrt(SWAP)^dag
    normal = np.cross(p2 - p1, p3 - p1)
    return np.dot(w-p1,normal) > 0

def in_region2(w):#w needs to be an np array
    """Check if a point is in the tetrahedron between CZ, I, sqrt(SWAP), sqrt(iSWAP)"""
    p1 = np.array([1/2,0,0]) #CZ/CNOT
    p2 = np.array([1/4,1/4,0]) #sqrt(iSWAP)
    p3 = np.array([1/4,1/4,1/4]) #sqrt(SWAP)
    normal = np.cross(p3-p1,p2-p1)
    return np.dot(w-p1,normal) > 0

def in_region3(w):#w needs to be an np array
    """Check if a point is in the tetrahedron above the PE polyhedron"""
    p1 = np.array([1/4,1/4,1/4])#sqrt(SWAP)
    p2 = np.array([3/4,1/4,1/4])#sqrt(SWAP)^dag
    p3 = np.array([1/2,1/2,0])#iSWAP
    normal = np.cross(p1-p3,p2-p3)
    return np.dot(w-p3,normal) > 0

def weyl_part(w):#w needs to be an np array
    """Check which part of the weyl chamber a pt belongs to"""
    if in_region1(w):
        return 1
    if in_region2(w):
        return 2
    if in_region3(w):
        return 3
    return 0 #in this case the point is a PE
