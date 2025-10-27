import numpy as np

def forward_substitution(L,b):
    """ Solve a system Lx=b,
    where L is a lower triangular coeffiecent matrix,
    and b is the right hand side vector,
    or a matrix where each column is a right hand side vector

    Parameters
    -------------
    L: array_like
        lower triangular matix, size = (n,n)
    b: array like
        Right had side(s), size = (n, ) or (n,m)
        where m is the number of right hand sides

    Returns
    ------------
    numpy.ndarray
        The vector or matix of solutions x.
        This will have the same shape as b.
    """
    # checking that inputs can be convered to array of floats
    # making a local copy of the arrays 
    L = np.array(L, dtype = float)
    b = np.array(b, dtype = float)
    # check shape of matrix L
    L_shape = L.shape
    if not len(L_shape) == 2:
        raise ValueError("coefficent matrix L has shape {L_shape},", {len(L_shape)},
            "Must be 2.")
    n = L_shape[0]
    if n != L_shape[1]:
         raise ValueError("coefficent matrix L has shape {L_shape,", {len(L_shape)},
          "Must be square.")
    # check shape of right hand side b
    

    return np.zeros_like(b)
