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
    xd = np.linspace(0,10,num=5)
    yd_lin = []
    yd_quad = []
    yd_cub = []
    i=0
    for x in xd:
        yd_lin.append(0)
        yd_lin[i] = (xd[i])*2
        yd_quad.append(0)
        yd_quad[i] = (xd[i])**2
        yd_cub.append(0)
        yd_cub[i] = (xd[i])**3
        i+=1

    n_test = 2.2
    o1 = spline_function(xd,yd_lin,order =1)  
    o2 = spline_function(xd,yd_lin,order =2)
    o3 = spline_function(xd,yd_lin,order =3)
    print (" xd:\n",xd,"\n yd: \n",yd_lin,)
    print(n_test,"*2, expected:",n_test *2,
          "\n order 1:",o1(n_test),
          "\n order 2:",o2(n_test),
          "\n order 3:",o3(n_test))
                      
    o1 = spline_function(xd,yd_quad,order =1)  
    o2 = spline_function(xd,yd_quad,order =2)
    o3 = spline_function(xd,yd_quad,order =3)
    print (" xd:\n",xd,"\n yd: \n",yd_quad)
    print(n_test,"^2, expected:",n_test**2,
          "\n order 1:",o1(n_test),
          "\n order 2:",o2(n_test),
          "\n order 3:",o3(n_test))
                      
    o1 = spline_function(xd,yd_cub,order =1)  
    o2 = spline_function(xd,yd_cub,order =2)
    o3 = spline_function(xd,yd_cub,order =3)
    print (" xd:\n",xd,"\n yd: \n",yd_cub)
    print(n_test,"^3, expected:",n_test**3 ,
          "\n order 1:",o1(n_test),
          "\n order 2:",o2(n_test),
          "\n order 3:",o3(n_test))

    xd = np.linspace (0,10,num = 15)
    yd = []
    i=0
    while i < len (xd):
        yd.append(0)
        yd[i] = 2*((xd[i])**4)
        i = i+1
    n_test = 3
    print (yd, xd)
    s_expected = UnivariateSpline(xd,yd,s=0,ext="raise", k =3)
    s_cub = spline_function(xd,yd,order=3)
    s_e = s_expected(n_test)
    s3 = s_cub(n_test)
    n_test_calc = (2*(n_test**4))
    print ("2*(",n_test,")^4  + 1.5* (",n_test,")^2  =",n_test_calc,
           "\n expected :", s_e,
           "\n calculated:",s3)
    
    
    


    
def main():
    print ("Test, I am in main().")
   test_gauss_iter_solve()
    test_spline_function()
if __name__ == "__main__":
    main()
