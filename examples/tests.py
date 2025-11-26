import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline 
from lab02.linalg import(
    gauss_iter_solve,
    spline_function
    )



def test_gauss_iter_solve():
    #test arrays
    A= np.array ([[4.0,1.0,2.0],
                  [3.0,5.0,1.0],
                  [1.0,1.0,3]])
    B = np.array ([4.0,7.0,3.0])

    X_seidel = gauss_iter_solve(A,B)
    X_jacobi = gauss_iter_solve(A,B,alg="jacobi")
    X_expected = np.linalg.solve(A,B)
    print (" Calculated X_seidel: \n",X_seidel,
            "\n Calculated X_jacobi: \n",X_jacobi,
           "\n Expected X: \n",X_expected)

    B = np.array ([[1.0,0.0,0.0],
                   [0.0,1.0,0.0],
                   [0.0,0.0,1.0]])
    Ainv_seidel = gauss_iter_solve(A,B)
    Ainv_jacobi = gauss_iter_solve(A,B,alg="jacobi")
    Ainv_expected = np.linalg.inv(A)
    print (" Calculated Ainv_seidel: \n",Ainv_seidel,
            "\n Calculated Ainv_jacobi: \n",Ainv_jacobi,
           "\n Expected Ainv: \n",Ainv_expected)
    I_seidel = np.matmul(A,Ainv_seidel)
    I_jacobi = np.matmul (A,Ainv_jacobi)
    I_expected = np.matmul (A,Ainv_expected)
    print (" Calculated I_seidel: \n",I_seidel,
            "\n Calculated I_jacobi: \n",I_jacobi,
           "\n Expected I: \n",I_expected)

def test_spline_function():
    xd = [1,2,3,4,5]
    yd = [5,10,15,20,25]
    o1 = spline_function(xd,yd,order =1)  
    o2 = spline_function(xd,yd,order =2)
    o3 = spline_function(xd,yd,order =3)
    print(o1(2.5))
    
def main():
#    print ("Test, I am in main().")
#   test_gauss_iter_solve()
    test_spline_function()
if __name__ == "__main__":
    main()
