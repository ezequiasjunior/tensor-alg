#!/usr/bin/python3
# -*- coding: utf-8 -*-
#-------------------------------------------------
# Module containing the implemented functions  
# during the Tensor Algebra course. 
# These functions are composing my own tensor 
# toolbox.
#-------------------------------------------------
## Author: Ezequias Júnior
## Version: 0.5.3
## Email: ezequiasjunio@gmail.com
## Status: in development

# Imports
import numpy as np


# Usefull functions
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


def mode_index(shape_id, mode):
    """Auxiliary function to permute the indices of the given array in such a 
    way that the first axis corresponds to the axis of the n-mode dimension.
    
    Parameters:
    -----------
    shape_id : [1-D array]
        Indices of the tensor shape vector.
    mode : [scalar]
        Selected fiber mode.
    
    Returns:
    --------
    [1-D array]
        Reordered axes list.
    """
    # Selected mode:
    if mode == 1:
        i = 2
    elif mode == 2:
        i = 1
    else:
        i = mode
    
    # Storing the first element:
    aux = shape_id[0]
    
    # Changing positions:
    shape_id[0] = shape_id[-i]
    shape_id[-i] = aux
    
    # Return the inplace permuted indices:
    return shape_id


def m_mode_prod_shape(tensor, matrix, mode):
    """Auxiliary function to calculate the new shape of the resulting tensor 
    for the mode product.
    
    Parameters:
    -----------
    tensor : [n-D array]
        Target tensor of the mode product.
    matrix : [2-D array]
        Matrix that is applied to the tensor.
    mode : [scalar]
        The selected mode.
    
    Returns:
    --------
    [1-D array]
        Vector containing the new shape.
    """

    # Taking the tensor shape:
    shape = np.asarray(tensor.shape)
    # Ordering in the notation I_1, ..., I_n:
    ord_shape = np.hstack([shape[-2:], shape[:-2][::-1]]) 
    # Assigning the new dimension as the number of rows of the matrix:
    ord_shape[mode - 1] = matrix.shape[0]
    # Returning the new shape in Numpy order
    return np.hstack([ord_shape[2:][::-1], ord_shape[:2]])


def tensor_norm(tensor):
    """Calculates the tensor norm by taking the square root of the 
    sum of the squares of all its elements.

    Parameters:
    -----------
    tensor : [n-D array]
        Target tensor

    Returns:
    --------
    [scalar]
        Tensor norm.
    """
    return np.sqrt(np.sum(np.abs(tensor)**2))


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


def kr(*args):    
    """Calculates the Khatri-Rao product between n matrices.
    
    Parameters:
    -----------
    *args : [2-D array]
        List of n matrices K x J.
    
    Returns:
    --------
    mt_out: [2-D array]
        Matrix K^{n} x J: The Khatri-Rao product between mt_a and mt_b.
    """
    # Function to calculate the product kr(kr(A, A), A)
    # Storing the number of columns:
    ncol = args[0].shape[1]
    # Testing the condition of existence of the Khatri-Rao product:
    assert ncol == args[1].shape[1],\
         f'The matrices must have the same number of columns!'
    
    kr_prod = args[0]
    for matrix in args[1:]:
        # Calculating the product: it is returned a 3-D array i x k x j
        # mt_out = np.einsum('ij, kj-> ikj', mt_a, mt_b)
        # to save some micro seconds:
        kr_prod = (kr_prod[:, np.newaxis, :] * matrix).reshape((-1, ncol))
        # Applying the reshape to get the expected dimension of the product:
        # (i x k, j)
    return kr_prod


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


# Tensor operations
def unfold(target, mode):
    """Function that extract the n-mode unfolding of the target tensor.
    
    Parameters:
    -----------
    target : [n-D array]
        Tensor to be unfolded.
    mode : [scalar]
        Selected mode.
    
    Returns:
    --------
    [2-D array]
        The unfolded matrix.
    """

    # Taking the tensor shape:
    shape = np.asarray(target.shape)
    # Ordering in the notation I_1, ..., I_n:
    ord_shape = np.hstack([shape[-2:], shape[:-2][::-1]])
    
    # Selecting the mode dimension to be the firt dimension of the numpy array:
    select_dim = mode_index(np.arange(shape.size), mode)
    # Transposing the tensor:
    fibers = target.transpose(*select_dim)
    
    # Returning the unfolded matrices in Kolda standard form: 
    if mode==3 or mode==2:
        return fibers.reshape(ord_shape[mode - 1], -1, order='F')
    else:
        return fibers.reshape(ord_shape[mode - 1], -1)


def fold(target, shape, mode):
    """Function that performs the mode fold operation of a target matrix to a 
    tensor with the given shape.
    
    Parameters:
    -----------
    target : [2-D array]
        Mode unfolded matrix.
    shape : [1-D array]
        Tensor shape vector
    mode : [scalar]
        Selected mode.
    
    Returns:
    --------
    [n-D array]
        Folded tensor.
    """

    # Taking the tensor shape:
    shape_arr = np.asarray(shape)
    # Selecting the mode dimension:
    select_dim = mode_index(np.arange(shape_arr.size), mode)
    # Reverting the reshape operation used in unfold:
    if mode == 3 or mode == 2:
        fibers = target.reshape(*shape_arr[select_dim], order='F')
    else: 
        fibers = target.reshape(*shape_arr[select_dim])
    # Returning the folded tensor reverting the traspose operation 
    # used in unfold:
    return fibers.transpose(*select_dim)


def m_mode_prod(tensor, mt_list, mode_list=None):
    """Functon that calculates the mode product of a tensor by a 
    list of matrices, applying the m-mode product to the m-th matrix.
    
    Parameters:
    -----------
    tensor : [n-D array]
        Target tensor.
    mt_list : [list]
        List of matrices.
    
    Returns:
    --------
    [n-D array]
        Resultant tensor of the multilinear product
    """
    
    # Listing the modes:
    if mode_list is not None:
        modes = np.asarray(mode_list) + 1
    else:
        modes = np.arange(len(mt_list)) + 1
    
    # Calculating the new shape:
    new_shape = m_mode_prod_shape(tensor, mt_list[0], modes[0])
    result = fold(mt_list[0] @ unfold(tensor, modes[0]), new_shape, modes[0])
    # Calculating the product for the remaining matrices:
    for mode, matrix in zip(modes[1:], mt_list[1:]):
        new_shape = m_mode_prod_shape(result, matrix, mode)
        result = fold(matrix @ unfold(result, mode), new_shape, mode)
    # Returning the resultant tensor:
    return result


# Tensor decompositions
def hosvd(tensor, rank_list=None):
        
    order = len(tensor.shape)
    mt_u = [None]*order
    mt_core = [None]*order
    
    # Multilinear rank approximation -> Truncated HOSVD:
    if rank_list is not None:
        if len(rank_list) == order:

            for i, mode in enumerate(np.arange(order) + 1):
                target = unfold(tensor, mode)
                u, _, _ = np.linalg.svd(target)
                mt_u[i] = u[:, :rank_list[i]] 
                mt_core[i] = u[:, :rank_list[i]].conj().T
    
        else: 
            raise Exception('the number of dimensions must be equal' +\
                            ' to the tensor order.')
    
    # Full rank approximation -> HOSVD: 
    else:    
        for i, mode in enumerate(np.arange(order) + 1):
            target = unfold(tensor, mode)
            u, _, _ = np.linalg.svd(target)
            mt_u[i] = u
            mt_core[i] = u.conj().T
        
    core_tensor = m_mode_prod(tensor, mt_core)    
    return core_tensor, mt_u


def hooi(tensor, eps=1e-4, num_iter=100, rank_list=None):
    
    order = len(tensor.shape)
    mt_a = [None]*order
    modes = np.arange(order)
    
    if rank_list is not None:
        # Initializing via Truncated HOSVD:
        core_aux, mt_u_aux = hosvd(tensor, rank_list)
        for k in range(num_iter):
            for i in range(order):
                # Matrices selection:
                mask = np.ones(modes.size, dtype=bool)
                mask[i] = False
                # Auxiliary tensor:
                mt_aux = [mt_u_aux[idx].conj().T 
                                       for idx in range(order) if mask[idx]]
                target = m_mode_prod(tensor, mt_aux, modes[mask])
                # Update matrices:
                u, _, _ = np.linalg.svd(unfold(target, modes[i] + 1))
                mt_a[i] = u[:, :rank_list[i]]
            # Construct the core tensor:
            mt_core = [a.conj().T for a in mt_a]
            core_tensor = m_mode_prod(tensor, mt_core)
            # Approximation:
            tensor_approx = m_mode_prod(core_tensor, mt_a)
            # Convergence w.r.t. the target tensor:
            error = tensor_norm(tensor - tensor_approx)/tensor_norm(tensor)
            if error <= eps:
                print(f'Number of iterations: {k+1}; Error: {error}')
                break
            else:
                core_aux = core_tensor
                mt_u_aux = mt_a

    else:
        # Initializing via HOSVD:
        core_aux, mt_u_aux = hosvd(tensor)
        for k in range(num_iter):
            for i in range(order):
                # Matrices selection:
                mask = np.ones(modes.size, dtype=bool)
                mask[i] = False
                # Auxiliary tensor:
                mt_aux = np.asarray([a.conj().T for a in mt_u_aux])
                target = m_mode_prod(tensor, mt_aux[mask], modes[mask])
                # Update matrices:
                u, _, _ = np.linalg.svd(unfold(target, modes[i] + 1))
                mt_a[i] = u
            # Construct the core tensor:
            mt_core = [a.conj().T for a in mt_a]
            core_tensor = m_mode_prod(tensor, mt_core)
            # Approximation:
            tensor_approx = m_mode_prod(core_tensor, mt_a)
            # Convergence w.r.t. the target tensor:
            error = tensor_norm(tensor - tensor_approx)/tensor_norm(tensor)
            if error <= eps:
                print(f'Number of iterations: {k+1}; Error: {error}')
                break
            else:
                core_aux = core_tensor
                mt_u_aux = mt_a
    
    return core_tensor, mt_a



# Special tensorized Matrix Factorizations
def mlskrf(mt_x, nrow_a, nrow_b):
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