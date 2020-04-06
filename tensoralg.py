#!/usr/bin/python3
# -*- coding: utf-8 -*-
#-------------------------------------------------
# Module containing the functions implemented 
# during the Tensor Algebra course. 
# These functions these functions compose my own 
# tensor toolbox.
#-------------------------------------------------
## Author: Ezequias Júnior
## Version: 0.4.2
## Email: ezequiasjunio@gmail.com
## Status: in development

# Imports
import numpy as np


# Utils
def vec(mt_x):
    """Applies the vec operator on matrix mt_x (M, N), returning a 
    column vector(M*N, 1) containing the columns of mt_x stacked. 
    
    Parameters:
    -----------
    mt_x : [2-D array]
        Matrix M x N.
    
    Returns:
    --------
    out: [2-D array]
        Column vector containing the columns of mt_x.
    """
    return mt_x.flatten(order='F').resape(-1, 1)


def unvec(vt_x, nrow, ncol):
    """Applies the unvec (nrow, ncol) operation on vector vt_x, returning a
    matrix of size nrow x ncol.
    
    Parameters:
    -----------
    vt_x : [2-D array]
        Column vector nrow*ncol x 1.
    nrow : [scalar]
        Number of rows.
    ncol : [scalar]
        Number of columns.
    
    Returns:
    --------
    out: [2-D array]
        Matrix X nrow x ncol. 
    """
    return vt_x.reshape(nrow, ncol, order='F')
    
def extract_block(blkmatrix, shape_a, shape_b):
    """Auxiliary function to extract the blocks of a block matrix constructed 
    by kron(A, B) (MPxNQ) by rearrangig the Kronecker product into a rank 1 
    matrix X_tilde.
    
    Parameters:
    -----------
    blkmatrix : [2-D array]
        Matrix X = kron(A, B).
    shape_a : [tuple]
        Size of matrix A.
    shape_b : [tuple]
        Size of matrix B.

    Returns:
    --------
    [2-D array]
        Matrix X_tilde with each vectorized block as columns.
    """
    # Extracting the dimensions of A and B:
    nrow_a, ncol_a = shape_a
    nrow_b, ncol_b = shape_b
    # Ordering the blocks in column-major order:
    split = np.array(np.hsplit(blkmatrix, ncol_a))
    # stacking the blocks in a 3rd order tensor:
    aux = split.reshape(nrow_a*ncol_a, *shape_b)
    # Reshaping into a matrix with each slice as columns (a_ij*vec(B)):    
    return aux.T.reshape(nrow_b*ncol_b, nrow_a*ncol_a)

# TODO:
# def my_unfold(tensor, mode): 
#     '''
#     unfolding of an 3rd order tensor.

#     :param tensor: matriz i x k
#     :param mode: modo desejado (1, 2 ou 3)
#     :return : matriz do modo {(i x jk), (j, ik), (k, ij)}
#     '''
#     t = tensor.shape
#     if mode is 1: # i, jk
#         return np.transpose(tensor, (1,0,2)).reshape(t[mode], int(np.prod(t)/t[mode]))
#     elif mode is 2: # j, ik
#         return np.transpose(tensor, (2,0,1)).reshape(t[mode], int(np.prod(t)/t[mode]))
#     elif mode is 3:# k, ij
#         # shape 3 ordem = (i0, i1, i2)
#         return np.transpose(tensor, (0,2,1)).reshape(t[0], int(np.prod(t)/t[0]))
#     else:
#         print("Modo não suportado")
#         return None


# Matrix products
def hadamard(mt_a, mt_b):
    """
    Calculates the element-wise product between the two matrices 
    mt_a and mt_b.

    Parameters:
    -----------
    mt_a : [2-D array]
        Matrix M x N.
    mt_b : [2-D array]
        Matrix M x N.

    Returns:
    --------
    out: [2-D array]
    Matrix M x N: The Hadamard product between mt_a and mt_b.
    """

    # Defining the subscripts for Einstein summation convention such that 
    # perform the following nested loop:
    # for i in range(mt_a.shape[0]):
    #     for j in range(mt_a.shape[1]):
    #         out[i, j] = mt_a[i, j]*mt_b[i, j] 
    subs = 'ij,ij->ij'
    
    return np.einsum(subs, mt_a, mt_b)


def kron(mt_a, mt_b):
    """Calculates the Kronecker Product between the two matrices 
    mt_a and mt_b.
    
    Parameters
    ----------
    mt_a : [2-D array]
        Matrix M x N.
    mt_b : [2-D array]
        Matrix I x J.
    
    Returns
    -------
    mt_out: [2-D array]
    Matrix MI x NJ: The Kronecker product between mt_a and mt_b.
    """

    # Storing the matrices dimensions:
    m, n = mt_a.shape
    i, j = mt_b.shape 

    # Computing the Kronecker product using Numpy broadcasting:
    # it is returned a 4-D array (M x I x  N x J)
    mt_out = mt_a[:, np.newaxis, :, np.newaxis] *\
             mt_b[np.newaxis, :, np.newaxis, :]

    # Applying the reshape to get the expected dimension of the 
    # product (MI x NJ):
    return mt_out.reshape(m*i, n*j)


def kr(mt_a, mt_b):
    """Calculates the Khatri-Rao product between the two matrices 
    mt_a and mt_b.
    
    Parameters
    ----------
    mt_a : [2-D array]
        Matrix I x J.
    mt_b : [2-D array]
        Matrix K x J.
    
    Returns
    -------
    mt_out: [2-D array]
    Matrix IJ x K: The Khatri-Rao product between mt_a and mt_b.
    """

    # Testing the condition of existence of the Khatri-Rao product:
    assert mt_a.shape[1] == mt_b.shape[1],\
         f'The matrices must have the same number of columns!'

    # Storing the number of columns:
    ncol = mt_a.shape[1]

    # Calculating the product: it is returned a 3-D array i x k x j
    # mt_out = np.einsum('ij, kj-> ikj', mt_a, mt_b)
    # to save some micro seconds:
    mt_out = mt_a[:, np.newaxis, :] * mt_b

    # Applying the reshape to get the expected dimension of the product (ixk,j):
    return mt_out.reshape((-1, ncol))


# Special Matrix Factorizations
def lskrf(mt_x, nrow_a, nrow_b):
    """Solving the problem: min ||X - kr(A, B)||^2 by estimating the matrices 
    A and B by the Least Squares Khatri-Rao Factorization method, where mt_x 
    was constructed following the model X = kr(A, B) with A of size (nrow_a, 
    ncol) and B of size (nrow_b, ncol) with P being the number of columns of X.

    Parameters:
    -----------
    mt_x : [2-D array]
        Input matrix (nrow_a*nrow_b x ncol) to be factorized.
    nrow_a : [scalar]
        Number of rows of matrix mt_a.
    nrow_b : [scalar]
        Number of rows of matrix mt_b.

    Returns:
    --------
    mt_a: [2-D array]
    mt_b: [2-D array]
        Estimated matrices mt_a and mt_b.
    """
    
    # Number of columns of mt_x:
    ncol = mt_x.shape[1]
    # Checking if mt_x is complex and alocating the estimated matrices:
    if np.iscomplexobj(mt_x):
        mt_a = np.zeros((nrow_a, ncol), dtype=np.complex_)
        mt_b = np.zeros((nrow_b, ncol), dtype=np.complex_)
    else: # real case:
        mt_a = np.zeros((nrow_a, ncol))
        mt_b = np.zeros((nrow_b, ncol))
    
    # Making a 3rd order tensor with the X_p matrices as forntal scilces
    # the number of slices is the number of columns ncol = p of mt_x:
    
    if np.isfortran(mt_x): # Dealing with MATLAB arrays (Fortram)
        target = mt_x.T.reshape(ncol, nrow_b, nrow_a, order='F')
    else: # 'A' = column-major indexes
        target = mt_x.T.reshape(ncol, nrow_b, nrow_a, order='A')
    
    # Calculating the SVD of each X_p matrix: U \Sigma V^{H}
    u, sigma, vh = np.linalg.svd(target)
    
    # Filling the columns of mt_a and mt_b with the respective rank-1 approx.:
    for p in range(ncol): 
        mt_a[:, p] = np.sqrt(sigma[p, 0]) * vh[p, 0, :]
        mt_b[:, p] = np.sqrt(sigma[p, 0]) * u[p, :, 0]

    return mt_a, mt_b


def lskronf(mt_x, shape_a, shape_b):
    """Solving the problem: min ||X - kron(A, B)||^2 by estimating the matrices 
    A and B by the Least Squares Kronecker Factorization method, where mt_x 
    was constructed following the model X = kron(A, B) with A of size (M, N) 
    and B of size (P, Q).

    Parameters:
    -----------
    mt_x : [2-D array]
        Input matrix (M*P x N*Q) to be factorized.
    nrow_a : [tuple]
        Size of matrix mt_a.
    nrow_b : [tuple]
        Size of matrix mt_b.

    Returns:
    --------
    mt_a: [2-D array]
    mt_b: [2-D array]
        Estimated matrices mt_a and mt_b.
    """

    # Constructing the rank-1 matrix X_tilde:
    target = extract_block(mt_x, shape_a, shape_b)

    # Calculating the SVD of X_tilde: U \Sigma V^{H}
    u, sigma, vh = np.linalg.svd(target)

    # Calculating mt_a and mt_b as the  best rank-1 approximation:
    mt_a = unvec(np.sqrt(sigma[0]) * vh[0, :], *shape_a)
    mt_b = unvec(np.sqrt(sigma[0]) * u[:, 0], *shape_b)   

    return mt_a, mt_b