import numpy as np

def main():
    print ("Test, I am in main().")

    test_forward_substitution()

def test_forward_substitution():
    #lower triangle coeffiecent matrix
    print ("testing forward substitution")
    A = np.array([
        [2135.0, 0.0, 0.0, 0.0],
        [-2135.0, 5200.0, 0.0, 0.0],
        [0.0, -5200.0, 5796.0, 0.0],
        [0.0, 0.0, -5796.0, 7060.0] ])
    # right hand side vector
    b = np.array([500.0,700.0,1000.0,500.0])

    #solve using numpy.linalg.solve()
    x_exp = np.linalg.solve(A,b)
    print ("expected x (from np.linalg.solve):\n",x_exp,
           "\nA:\n",A,
           "\nb:\n",b)

if __name__ == "__main__":
    main()
