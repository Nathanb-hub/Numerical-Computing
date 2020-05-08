import numpy as np

mat = np.matrix([[1,2,3],[4,5,6]])
print(mat)
print(mat.shape)

print(np.matrix('1 2;3 4;5 6'))


"""
Identité d'une matrice
"""

a = np.identity(3) # matrice unité à 3 dimensions
b = np.eye(2)      # matrice unité à 2 dimensions
c = np.eye(2,4,3) # 1e param est le nbre de ligne, le 2 est les colonnes, 3e est index

print(a)
print(b)
print(c)



"""
 Matrice Triangulaire
"""
a = np.tri(2)
b = np.tri(3,5,1)
print(a)
print(b)


 """
 Conversion array 1D en une matrice diagonale
 """

a = np.array([4,0,-2])
b = np.diag(a)

print(b)



 """
 Operation sur les vecteurs
 """
u = np.array([1,2,3])
v = np.array([4,5,6])

print(2*u + v)
#print(norm(u))
print(np.vdot(u,v)) # vdot calcule le produit scalaire (dot product)
print(np.cross(u,v))# calcule le produit vectoriel (cross product)
print(u*v)


 """
 Propriété des matrices
 """

A = np.matrix('1 2 3;4 5 6')
B = np.identity(2)
C = np.matrix('1 2 3;2 4 6;3 6 9')

print(A.shape)
print(np.diag(A))
print(np.trace(B))
print(matrix_rank(A))
print(det(C))

A = np.matrix('1 2 3;4 5 6')
B = np.identity(2)

print(2 * A[:,:2] + B)
print(A.T) # .T donne la trasnposée de la matrice

C = np.diag(np.array([1,2,3]))
D = np.matrix('1 2 3;2 4 6;3 6 9')
E = np.mat(np.array([[1,2],[3,4]]))


print(D* C[:,:2])
print(E.I) # .I donne la matrice inverse


