#!/usr/bin/python3
# -*- coding: utf-8 -*-
#-------------------------------------------------
# Module containing the implemented functions  
# during the Tensor Algebra course. 
# These functions are composing my own tensor 
# toolbox.
#-------------------------------------------------
## Author: Ezequias Júnior
## Version: 1.0.2
## Email: ezequiasjunio@gmail.com
## Status: in development.


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
    return mt_x.flatten(order='F').reshape(-1, 1)


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


def npy2math(shape):
    """Auxiliary function for convert an numpy array shape to the Octave 
    notation.

    Parameters:
    -----------
    shape : [tuple/list/array]
        Input shape for conversion.

    Returns:
    --------
    [list]
        Converted shape.
    """
    shape = list(shape)
    # Ordering tensor shape in the notation I_1, ..., I_n:
    math_order = shape[-2:] + shape[:-2][::-1]
    return math_order


def math2npy(shape):
    """Auxiliary function for convert an array shape to the numpy notation.

    Parameters:
    -----------
    shape : [tuple/list/array]
        Input shape for conversion.

    Returns:
    --------
    [list]
        Converted shape.
    """
    shape = list(shape)
    # Ordering tensor shape in the Numpy notation:
    npy_order = shape[2:][::-1] + shape[:2]
    return npy_order


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


def kron(*args):
    """Calculates the Kronecker Product between n matrices.
    
    Parameters:
    -----------
    *args : [2-D array]
        List of n matrices K x J.
    
    Returns:
    --------
    mt_out: [2-D array]
        Matrix K^{n} x J^{n}: The Kronecker product.
    """
    # Auxiliary function:
    def nkron(mt_a, mt_b):
        # Function to calculate the product kron(A, B)
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
    
    # Calculates the product kron(...kron(kron(A0, A1),...), An)
    kron_prod = args[0]
    for matrix in args[1:]:
        kron_prod = nkron(kron_prod, matrix)
    
    return kron_prod
 

def kr(*args):
    """Calculates the Khatri-Rao product between n matrices.
    
    Parameters:
    -----------
    *args : [2-D array]
        List of n matrices K x J.
    
    Returns:
    --------
    mt_out: [2-D array]
        Matrix K^{n} x J: The Khatri-Rao product.
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
    ncol) and B of size (nrow_b, ncol).

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
        Selected mode. Supported modes: {1, 2, 3}.
    
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


def tensor_vec(tensor):
    """Vectorize a tensor os shape (I_1 x ... x I_n) into a vector of shape 
    (I_1...I_n x 1).

    Parameters:
    -----------
    tensor : [N-D array]
        Tensor to be vectorized.

    Returns:
    --------
    [2-D array]
        Column vector.
    """
    # Permute dimensions:
    target = np.transpose(tensor, npy2math(np.arange(tensor.ndim)))
    # Select 1-mode fibers:
    target_m1 = target.reshape(target.shape[0], -1, order='F') 
    # Return the vectorization of 1-mode fibers:
    return vec(target_m1)


def tensor_unvec(vector, math_shape):
    """Reshape a vector of shape (I_1...I_n x 1) into a tensor of math_shape.

    Parameters:
    -----------
    vector : [2-D array]
        Input vector.

    Returns:
    --------
    [N-D array]
        Reshaped tensor.
    """
    # Reshape the vector unpacking the 1-mode fibers:
    target_m1 = vector.reshape(math_shape[0], -1, order='F')
    # Reshape to the tensor shape:
    target = target_m1.reshape(math_shape, order='F')
    # Returning the tensor permuted to the numpy order:
    return target.transpose(math2npy(np.arange(target.ndim)))


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
    mode_list: [list]
        List of modes to apply the product. Default: None.    
    Returns:
    --------
    [n-D array]
        Resultant tensor of the multilinear product.
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
    rshape = np.array(result.shape) # Fix: considering vector n mode product.
    return result.reshape(rshape[rshape > 1])


def tensor_kron(*args):
    """Calculates the Kronecker Product between n tensors.
    
    Parameters:
    -----------
    *args : [N-D array]
        List of n tensors.
    
    Returns:
    --------
    mt_out: [N-D array]
        The tensor Kronecker product.
    """
    def tkron(ten_a, ten_b):
        # Auxiliary function to perform the tensor kronecker product:
        k = max(ten_a.ndim, ten_b.ndim)
        m_a = npy2math(ten_a.shape)
        m_b = npy2math(ten_b.shape)
        
        math_shape = m_b + m_a
        new_shape = np.array(m_a) * np.array(m_b)
        
        permutation = []
        for p in np.arange(k):
            permutation += [p, p + k]
        
        prod = kron(tensor_vec(ten_a), tensor_vec(ten_b))

        reshaped = prod.reshape(math_shape, order='F')
        
        permuted = reshaped.transpose(permutation)
        
        target = permuted.reshape(new_shape, order='F')
        
        return target.transpose(math2npy(np.arange(len(new_shape))))
        
    # Calculates the product kron(...kron(kron(T0, T1),...), Tn)
    kron_prod = args[0]
    for tensor in args[1:]:
        kron_prod = tkron(kron_prod, tensor)
    
    return kron_prod


# Tensor decompositions
def hosvd(tensor, rank_list=None):
    """Calculates the High Order Singular Value Decomposition of the given 
    tensor and allows the truncation for the R1,..., R_N rank approximation.

    Parameters:
    -----------
    tensor : [n-D array]
        Tensor to apply the decomposition.
    rank_list : [1-D array], optional
        List of new dimensions R_1,...,R_N, by default None.

    Returns:
    --------
    [n-D array, list]
        Core tensor and N factor matrices.
    """
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
    # Construct the core tensor:
    core_tensor = m_mode_prod(tensor, mt_core)    
    return core_tensor, mt_u


def hooi(tensor, eps=1e-4, num_iter=100, verb=False, rank_list=None):
    """Calculates the High Order Singular Value Decomposition of the given 
    tensor and allows the truncation for the R1,..., R_N rank approximation 
    using the High Order Orthogonal Iterations algorithm initialized with the 
    HOSVD.

    Parameters:
    -----------
    tensor : [n-D array]
        Tensor to apply the decomposition.
    eps : [scalar]
        Error criteria for the HOOI algorithm, by default 1e-4.
    num_iter : [scalar]
        Number of iterations for the HOOI algorithm, by default 100.
    verb : [boolean]
        Flag for allow verbose.
    rank_list : [1-D array], optional
        List of new dimensions R_1,...,R_N, by default None.

    Returns:
    --------
    [n-D array, list]
        Core tensor and N factor matrices.
    """
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
                if verb: print(f'Number of iterations: {k+1}; Error: {error}')
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
                if verb: print(f'Number of iterations: {k+1}; Error: {error}')
                break
            else:
                core_aux = core_tensor
                mt_u_aux = mt_a
    return core_tensor, mt_a


# Auxiliary functions:
def ord_unfold(tensor, mode):
    """Function that extract the n-mode unfolding of the target tensor. 
    Correction of the fibers order applyied.
    
    Parameters:
    -----------
    target : [n-D array]
        Tensor to be unfolded.
    mode : [scalar]
        Selected mode. Supported modes: {1, 2, 3}.
    
    Returns:
    --------
    [2-D array]
        The unfolded matrix.
    """
    shape = npy2math(tensor.shape)
    size = tensor.size

    axes = math2npy(np.arange(tensor.ndim))
        
    aux = np.arange(size).reshape(shape, order='F').transpose(axes)
        
    ord_idx = np.argsort(unfold(aux, mode)[0, :]) 
        
    unfolded = unfold(tensor, mode)

    return unfolded[:,ord_idx]

def ord_fold(ord_matrix, shape, mode):
    """Function that performs the mode fold operation of a target matrix to a 
    tensor with the given shape. Correction of the fibers order applyied.
    
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
    size = np.prod(shape)
    mshape = npy2math(shape)
    axes = math2npy(np.arange(len(shape)))
    ten_aux = np.arange(size).reshape(mshape, order='F').transpose(axes)
        
    vec_aux = tensor_vec(fold(ord_unfold(ten_aux, mode), shape, mode)).ravel()
    ord_idx = np.argsort(vec_aux)

    vector = tensor_vec(fold(ord_matrix, shape, mode))[ord_idx]
        
    tensor = vector.reshape(mshape, order='F').transpose(axes)
    return tensor


def cp_decomp(tensor, rank, eps=1e-6, num_iter=500, init_svd=True, verb=False):
    """CANDECOMP/PARAFAC decomposition of a tensor(I_1 x ... x I_n) with 
    respect to rank R in a set of N factor matrices A_n (I_n x R) throug the 
    Alternated Least Squares algorithm.

    Parameters:
    -----------
    tensor : [N-D array]
        Tensor to be factorized.
    rank : [scalar]
        R = rank, rank of the tensor/number of rank-1 components that 
        recontruct X.
    eps : [scalar], optional
        Tolerance for the error between estimatives of each iteration, 
        by default 1e-6.
    num_iter : [int], optional
        Number of iterations considered in the ALS algorithm, by default 200.
    init_svd : [bool], optional
        Flag for HOSVD initialization of the factor matrices.
    verb : [bool], optional
        Flag for verbose and to return the errors, by default False.

    Returns:
    --------
    list of [2-D array]
        Factor matrices after ALS convergence.
    """
    # Taking the tensor shape for the rows of each factor matrix:
    shape = np.asarray(tensor.shape)
    # Ordering in the notation I_1, ..., I_n:
    rows = np.hstack([shape[-2:], shape[:-2][::-1]]) 
    num_matrices = len(rows)
    # Initializing the N factor matrices:
    if init_svd:
        _, factor_mtx = hosvd(tensor, [rank]*num_matrices)
    elif np.iscomplexobj(tensor):
        factor_mtx = [np.random.rand(i, 2*rank).view(complex) for i in rows]
    else:
        factor_mtx = [np.random.rand(i, rank) for i in rows]
    # Storing the error between each iteration:
    error = np.zeros((num_iter, 1)) 
    # 1-mode unfolding of tensor:
    x1_aux = ord_unfold(tensor, 1)
    # Alternated Least Squares:
    for k in range(num_iter):
        # Estimating factor matrices
        for n in range(num_matrices):
            mask = np.ones(num_matrices, dtype=bool)
            mask[n] = False
            # N-1 selected matrices:
            mt_aux = [factor_mtx[m] for m in range(num_matrices) if mask[m]]
            # Estimating the n-th factor matrix:
            factor_mtx[n] = ord_unfold(tensor, n+1) @\
                            np.linalg.pinv(kr(*mt_aux[::-1]).T)
        # Reconstructing tensor 1-mode matrix:
        x1_hat = factor_mtx[0] @ kr(*factor_mtx[1:][::-1]).T
        # Calculating the error using the cost function:
        error[k] = np.linalg.norm(x1_aux - x1_hat, 'fro')**2
        # Convergence treatment:
        if np.abs(error[k] - error[k - 1]) < eps:
            if verb:
                print(f'Algorithm converged with {k+1} iterations and '+\
                      f'an achieved error of {error[k]} between iteractions.')
            break
        # Max. iterations check:
        elif k+1 == num_iter:
            if verb:
                print(f'Reached max. number of iterations ({k+1})! '+\
                      f'Current error: {error[k]}.')
    # End
    if not verb:
        return factor_mtx
    else:
        return factor_mtx, error


def cpd_tensor(factor_mtx):
    """Construct a N-th order tensor following the PARAFAC model using a 
    list of N factor matrices.

    Parameters:
    -----------
    factor_mtx : list [2-D array]
        List of N factor matrices (I_n, R).

    Returns:
    --------
    tensor : [N-D array]
        Reconstructed tensor (I_1, ..., I_N).
    """

    # Extract the size of each dimension of the reconstructed tensor:
    rows = [m.shape[0] for m in factor_mtx]
    # Ordering shapes in the Numpy notation:
    ord_shape = rows[2:][::-1] + rows[:2]
    # 1-Mode tensor folding: 
    x1 = factor_mtx[0] @ kr(*factor_mtx[1:][::-1]).T
    tensor = ord_fold(x1, ord_shape, 1)
    return tensor

# TODO: implement
def tkpsvd(tensor, shapes):
    # Perform the TKPSVD of a tensor into a d-number of shapes.
    # real tensor, math shapes
    def pd_input(tensor, shapes):
        # Reshape - Permute - Reshape (grouping indexes) process.
        deg = len(shapes)
        ordr = len(shapes[0])
    
        target = tensor_vec(tensor)
        # 1st reshape    
        shape1 = [shapes[i][j] for j in range(ordr) for i in range(deg)]
        reshaped = target.reshape(shape1, order='F')
        # Permutation
        perm_idx = np.arange(len(shape1)).reshape(deg, ordr, order='F')
        permutation = [perm_idx[i,j] for i in range(deg) for j in range(ordr)]
        permuted = reshaped.transpose(permutation)
        # 2nd reshape
        aux = np.array(permuted.shape)
        shape2 = [np.prod(aux[0 + ordr*i:ordr*(i + 1):1]) for i in range(deg)]
        pd_tensor = np.reshape(permuted, shape2, order='F')
        # Returned A_tilde degree-order tensor
        return pd_tensor.transpose(math2npy(np.arange(pd_tensor.ndim)))

    degree = len(shapes)
    order = len(shapes[0])
    target = pd_input(tensor, shapes)
    
    # PD process- Changed to PARAFAC
    num_terms = degree - 2
    matrices = cp_decomp(target, rank=num_terms, eps=1e-16, 
                                 num_iter=2000, init_svd=False)
    # CP indexes, the j terms: 
    rank1_terms = np.tile(np.arange(num_terms), 
                          degree).reshape(num_terms, degree, order='F')
    # Storing
    factors = [[None]*num_terms for i in range(degree)]
    sigma = np.zeros(num_terms)
    
    for j, select_term in enumerate(rank1_terms): 
        # sigmas 1 taking the norm of the columns of the cpd factor matrices A
        sigma[j] = 1
        for d in range(degree):
            column = matrices[d][:, [select_term[d]]]
            sigma_d = np.linalg.norm(column) 
            sigma[j] *= sigma_d
            factors[d][j] = tensor_unvec(column/sigma_d, shapes[d])

    # ------------------- USING HOSVD ------------------------------------------
    # core, matrices = hosvd(target)

    # # selected = core.transpose(npy2math(np.arange(core.ndim)))
    # selected = core

    # num_terms = core.ndim - 1 # d - 1, hosvd of d-order tensor 
    # rank1_terms = np.tile(np.arange(num_terms), 
    #                       degree).reshape(num_terms, degree, order='F')
    
    # # Storage:
    # sigma = np.zeros(num_terms)
    # sigmar = np.zeros(num_terms)

    # factors = [[None]*num_terms for i in range(degree)]
    # for j, select_term in enumerate(rank1_terms): 
    #     # nroot(s[1,1,1,]) colum[d][:,1]
    #     sigma[j] = np.power(np.abs(selected[tuple(select_term)]), 1/degree)
    #     sigmar[j] = selected[tuple(select_term)] # real sigmas
    #     for d in range(degree):
    #         factors[d][j] = tensor_unvec(matrices[d][:, [select_term[d]]], 
    #                                      shapes[d])
    #---------------------------------------------------------------------------
    return sigma, factors


def tkp_tensor(sigmas, factors, num_terms=None, num_factors=None):
    """Function to construct a N-order tensor following the TKPSVD 
    decomposition model.

    Parameters:
    -----------
    sigmas : [1-D array]
        Kronecker singular values.
    factors : [list]
        List containing de j rank-1 factors for each degree-tensors.
    num_terms : [int], optional
        Selecting the number of sum terms, by default None.
    num_factors : [int], optional
        Selecting the number of d-tensor factors, by default None.

    Returns:
    --------
    [2-D array, N-D array]
        Shapes of each factor and the reconstructed tensor.
    """
    if num_terms is None:
         num_terms = sigmas.size
    
    if num_factors is None:
         num_factors = len(factors)
    # Unpacking the factors into j terms {j {d1, d2, ...,dn  }}
    unpack = [[factors[d][j] for d in range(num_factors)] 
                                 for j in range(num_terms)]
    
    shapes = np.array([unpack[0][d].shape for d in range(num_factors)])
    
    sum_rank1_terms = np.zeros(shapes.prod(axis=0))
    for j in range(num_terms):
        sum_rank1_terms += sigmas[j] * tensor_kron(*unpack[j][::-1])
    
    return shapes, sum_rank1_terms


# Special tensorized Matrix Factorizations
def mlskrf(mt_x, nrow_a, use_hooi=False):
    """Solving the problem: min ||X - kr(A_1,..., A_N)||^2 by estimating the 
    matrices A_n by the Multidimensionla Least Squares Khatri-Rao Factorization 
    method, where mt_x was constructed following the model X = kr(A_1,..., A_n) 
    with each matrix A_n of size (nrows_a[n], ncol).

    Parameters:
    -----------
    mt_x : [2-D array]
        Input matrix (nrow_a[1]*...*nrow_a[N], x ncol) to be factorized.
    nrow_a : [1-D array]
        Number of rows of matrix each matrix A_n.
    use_hooi : [boolean]
        Flag to use HOOI algorithm to n-rank approximation.

    Returns:
    --------
    mt_a: list of [2-D array]
        Estimated matrices A_n.
    """
    assert mt_x.shape[0] == np.asarray(nrow_a).prod(),\
        f'The number of rows of mt_x must be equal to the product of nrow_a!'

    # Number of columns of mt_x:
    ncol = mt_x.shape[1]
    # Alocating the estimated matrices:
    num_matrices = len(nrow_a)
    mt_a = [np.zeros((row, ncol), dtype=np.complex_) for row in nrow_a]
    # Ordering shapes in the Numpy notation:
    ord_shape = nrow_a[::-1][2:][::-1] + nrow_a[::-1][:2]
    for r in range(ncol):
    # Making a n-th order rank-1 tensor using r-th column of mt_x 
    # the shape is the number of rows of each matrix (nrow_a[N]x...xnrow_a[1]):
        target = mt_x[:, r].reshape(*nrow_a).swapaxes(num_matrices-2, 
                                                      num_matrices-1)
        target = target.reshape(*ord_shape)
        # Taking best rank-1 approximation:
        if not use_hooi: 
            core, u = hosvd(target)
        else:
            core, u = hooi(target)
        
        for n in range(num_matrices):
        # Filling the columns of mt_a[n] with the respective n_rank-1 approx.:
            mt_a[n][:, r] = u[::-1][n][:, 0]
            mt_a[n][:, r] *= np.power(core.flat[0] + 0j, 1/num_matrices)

    return mt_a


def lskronf_3d(mt_x, shapes, use_hooi=False):
    """Solving the problem: min ||X - kron(A_1, A_2, A_3)||^2 by estimating the 
    matrices A_n by the Multilinear Least Squares Kronecker Factorization 
    method, where mt_x was constructed following the model X = kron(A_1, A_2, 
    A_3) with A_n of size shapes[n].

    Parameters:
    -----------
    mt_x : [2-D array]
        Input matrix to be factorized.
    nrow_a : [list]
        Sizes of each matrix A_n.
    use_hooi : [boolean]
        Flag to use HOOI algorithm to n-rank approximation.

    Returns:
    --------
    mt_a: list of [2-D array]
        Estimated matrices A_n.
    """
    # Auxiliary function for the rearrangment of the input matrix:
    def extract_block_3d(blkmatrix, shapes):    
        # Extracting the 1st dimensions:
        nrow_a, ncol_a = shapes[0]
        nrow_b, ncol_b = [int(x/y) for x, y in zip(blkmatrix.shape, shapes[0])] 
        # Ordering the outer blocks in row-order:
        split = np.array(np.hsplit(blkmatrix, ncol_a))
        # stacking the blocks in a 3rd order tensor:
        aux = split.reshape(nrow_a*ncol_a, nrow_b, ncol_b)
        nrow_a, ncol_a = shapes[1]
        nrow_b, ncol_b = [int(x/y) for x,y in zip((nrow_b, ncol_b), shapes[1])]
        # Allocating the matrix X_bar:
        if np.iscomplexobj(blkmatrix):
            out_x = np.zeros((nrow_b*ncol_b*nrow_a*ncol_a, np.prod(shapes[0])), 
                              dtype=np.complex_)
        else:
            out_x = np.zeros((nrow_b*ncol_b*nrow_a*ncol_a, np.prod(shapes[0])))
        # Access the inner blocks:
        for s in range(aux.shape[0]):
            split_aux = np.array(np.hsplit(aux[s], ncol_a))
            split_aux = split_aux.reshape(nrow_a*ncol_a, nrow_b, ncol_b).T
            # Column equals to vec(X_bar)
            out_x[:, [s]] = vec(split_aux.reshape(nrow_b*ncol_b,nrow_a*ncol_a))
        # After construct the matrix X_bar:    
        return out_x

    # Listing the size of each matrix for the tensor construction:
    size_aux = [s[0]*s[1] for s in shapes]
    # Alocating the estimated matrices:
    num_matrices = len(shapes)
    mt_a = [None]*num_matrices
    # Constructing the matrix X_bar rearranging the elements:
    rearrange_x = extract_block_3d(mt_x, shapes)
    vector_x = vec(rearrange_x)
    # Construct the tensor to apply the decomposition:
    # Ordering shapes in the Numpy notation:
    target = vector_x.reshape(*size_aux).swapaxes(num_matrices-2, 
                                                  num_matrices-1)
    # Taking the approximation:
    if not use_hooi: 
        core, u = hosvd(target)
    else:
        core, u = hooi(target)
    # Calculating mt_a as the best rank-1 approximation:
    for n, shape in enumerate(shapes):
        mt_a[n] = unvec(u[::-1][n][:, 0], *shape) + 0j
        mt_a[n] *= np.power(core.flat[0] + 0j, 1/num_matrices)
    
    return mt_a
