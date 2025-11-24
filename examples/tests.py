import numpy as np

from lab02.linalg import(
    gauss_iter_solve
    )



def test_gauss_iter_solve():
    A= np.array ([[4.0,1.0,2.0],
                  [3.0,5.0,1.0],
                  [1.0,1.0,3]])
    B = np.array ([4.0,7.0,3.0])

    X = gauss_iter_solve(A,B)
    X_expected = np.linalg.solve(A,B)
    print (" Calculated X: \n",X ,"\n Expected X: \n",X_expected)

    B = np.array ([[1.0,0.0,0.0],
                   [0.0,1.0,0.0],
                   [0.0,0.0,1.0]])
    X = gauss_iter_solve(A,B)
    print (X)
    I = np.matmul(A,X)
    print (I)

    
def main():
    print ("Test, I am in main().")
    test_gauss_iter_solve()
if __name__ == "__main__":
    main()
