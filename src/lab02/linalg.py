import numpy as np

def gauss_iter_solve(A,b,x0=None,tol=1e-8,alg="seidel"):
    """ Solve a system Ax=bn using Gauss-Seidel method

    Parameters
    -------------
    A: array_like
        Coefficent matrix, size = (n,n)
    b: array like
        Right had side(s), size = (n, ) or (n,m)
        where m is the number of right hand sides
    x0: array like
        (optional) Initial guess
        Default is none
    tol: float
        (optional) stopping condition
        default is a value of 1e-8
    alg: string
        (optional) flag for algorithim used
        defult is 'seidel', 'seidel' or 'jacobi' can be used

    Returns
    ------------
    x: array like 
        numpy.ndarray
        The vector or matix of solutions x.
        This will have the same shape as b.
    """
    # checking that inputs can be convered to array of floats
    # making a local copy of the arrays 
    A = np.array(A, dtype = float)
    b = np.array(b, dtype = float)
    # check shape of matrix A
    A_shape = A.shape
    if not len(A_shape) == 2:
        raise ValueError("coefficent matrix A has shape {A_shape},", {len(A_shape)},
            "Must be 2.")
    n = A_shape[0]
    if n != A_shape[1]:
         raise ValueError("coefficent matrix A has shape {A_shape},", {len(A_shape)},
          "Must be square.")
    # check shape of right hand side b
    b_shape = b.shape
    if len(b_shape) <1 or len(b_shape) >2:
        raise ValueError ("has dimenstion {len(b_shape)}. Must be 1 or 2")
    if n != b_shape[0]:
        raise ValueError ("b has leading dimension {len(b_shape[0])}, must match leading dimention of A ({n})")

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.array(x0)
        x_shape = x.shape
        if x_shape != b_shape:
            raise ValueError ("x0 ({x0}) does not have the same shape as b ({b})")
        elif x_shape[0] != b_shape[0] or x_shape[0] != n:
            raise ValueError ("x0 does not have the same number of rows as A or b")

    alg_input = alg.strip().lower()
    if alg_input not in ("seidel","jacobi"):
        raise ValueError ("algorithm must be 'seidel' or 'jacobi'")

    A_diag_inv = np.diag(1 / np.diag(A))
    A_star = A_diag_inv @ A

    id = np.identity(len(A))
    A_s_star = A_star - id

    b_star = A_diag_inv @ b
    
    # Do the Gauss Seidel and Jacobi algorithms
    # First initialize the counters and set maximum iterations
    n = 0
    n_max = 100
    eps_a = 2 * tol

    # The jacobi algorithm
    if alg_input == 'jacobi':
        while n < n_max and eps_a > tol:
            x_copy = x.copy()       # Create a copy of the initial guess 
            x = b_star - (A_s_star @ x)         # Do the algorithm
            dx = x - x_copy             # Calculate the difference between iterations
            n += 1      
            eps_a = np.linalg.norm(dx) / np.linalg.norm(x)      # Update the approx relative error

    # The Gauss-seidel algorithm
    else:

        k=0
        eps_a = 1
        while n_max > k and eps_a > tol:
            x_new = np.copy(x)
            print("step",k, "\n Eps_a",eps_a)
            for i in range(A_shape[0]):
                sum1 = np.dot(A[i, :i], x_new[:i])
                sum2 = np.dot(A[i,i+1:], x[i+1:])
                x_new[i] = (b[i] - sum1 - sum2 )/ A[i,i]

            eps_a = np.linalg.norm(x_new - x)
            print (eps_a)
            x = x_new
            k+=1


    return x
    
"""A_test = np.array([[-11,4],[7,-9]])
b_test = np.array ([-5,3])
x0_test = np.array ([1,1])

test = gauss_iter_solve(A_test,b_test)
print (test)"""

def spline_function(xd,yd,order=3):
    """
    parameters
    -----------------
    xd: array like
        array of float data of increasing values
    yd: array like
        array of float data with the same shape as xd
    order: int
        (optional) order of polynomial, default is 3

    returns 
    -----------------
    Function that takes x parameter and interpolates y values"""

    xd= np.array(xd,dtype = float).flatten()
    yd = np.array (yd,dtype = float).flatten()
    y_out =[]

    if len(xd) != len(yd):
        raise ValueError ("xd and yd are diffent lenghts, must be the same")
    if len (xd) != len(np.unique(xd)):
        raise ValueError ("input variable 'xd' has repeated values")
    if order not in (1,2,3):
        raise ("input order is not in accepted range(1,2,3)")

    x_min = xd[0]
    x_max = xd[-1]
    def f(x):

        x_arr = np.atleast_1d(x)
        if np.any((x_arr<x_min)|(x_arr>x_max)):
            raise ValueError(f"Input is out of range.({x_min}-{x_max})")
        n = len(xd)                    
        for xi in x_arr:
            idx = np.searchsorted(xd,xi)-1
            if idx<0:
                idx = 0
            if idx >= n -1:
                idx = n -2

            #Linear need 2 points
            if order == 1:
                x0 = xd[idx]
                x1 = xd[idx+1]
                y0 = yd[idx]
                y1 = yd[idx+1]
                yi = y0 +(y1-y0) * (xi-x0) / (x1-x0)

            #Quadratic need 3 points
            if order ==2:
                if idx == 0:
                    pts = [0,1,2]
                elif idx >= n -2:
                     pts = [n-3,n-2,n-1]
                else:
                    pts = [idx-1,idx,idx+1]
                xpts = xd[pts]
                ypts = yd[pts]

                yi = 0
                for i in range(3):
                    li = np.prod([(xi-xpts[j])/(xpts[i]-xpts[j]) for j in range(3) if j!=i])
                    yi += ypts[i]*li
                
            # cubic need 4 points 
            if order == 3:
                if idx <1:
                    pts = [0,1,2,3]
                elif idx > n-3:
                    pts = [n-4,n-3,n-2,n-1]
                else:
                    pts = [idx -1,idx,idx+1,idx+2]
                xpts = xd[pts]
                ypts = yd[pts]
                yi = 0
                for i in range(4):
                    li = np.prod([(xi-xpts[j])/(xpts[i]-xpts[j]) for j in range(4) if j!=i])
                    yi += ypts[i]*li
            y_out.append(yi)

        return y_out[0] if np.isscalar(x) else np.array(y_out)

    return f 
