#!/usr/bin/python3
import numpy as np


def hd_product(mt_a, mt_b):
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
    mt_out = mt_a[:, None, :, None] * mt_b[None, :, None, :]

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
    mt_out = np.einsum('ij, kj-> ikj', mt_a, mt_b)

    # Applying the reshape to get the expected dimension of the product (ixk,j):
    return mt_out.reshape((-1, ncol))
