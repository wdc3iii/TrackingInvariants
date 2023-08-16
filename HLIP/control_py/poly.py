import numpy as np

class Poly:

    def __init__(self, X:np.ndarray, Y:np.ndarray, D:np.ndarray):
        n = X.shape[0]
        A = np.zeros((n,n))
        for ii in range(n):
            di = D[ii]
            xi = X[ii]

            for k in range(di, n):
                A[ii, k] = self.factorial(k) / self.factorial(k - di) * pow(xi, k - di)

        self.coeffs = np.linalg.solve(A, Y)
    
    def factorial(self, k:int) -> int:
        if k <= 1:
            return 1
        elif k == 2:
            return 2
        elif k == 3:
            return 6
        elif k == 4:
            return 24
        elif k == 5:
            return 120
        elif k == 6:
            return 720
        else:
            return k * self.factorial(k - 1)
        
    def evalPoly(self, x, d) -> float:
        n = self.coeffs.shape[0]
        res = 0
        for ii in range(d, n):
            res += self.coeffs[ii] * self.factorial(ii) / self.factorial(ii - d) * pow(x, ii - d)
        return res

if __name__ == "__main__":
    x_swf_pos_z = np.array([0, 0.3, 0.7 * 0.3, 0, 0.4])
    y_swf_pos_z = np.array([0, 0, 0.07, 0.01, -0.05])
    d_swf_pos_z = np.array([0, 0, 0, 1, 1])
    # x_swf_pos_z = np.array([0, 0.4, 0.7 * 0.4, 0, 0.4, 0.7 * 0.4])
    # y_swf_pos_z = np.array([0, 0, 0.1, 0.05, -0.05, 0])
    # d_swf_pos_z = np.array([0, 0, 0, 1, 1, 1])
    p_swf_pos_z = Poly(x_swf_pos_z, y_swf_pos_z, d_swf_pos_z)

    t = np.linspace(0, 0.3, 1000)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t, [p_swf_pos_z.evalPoly(t_phase, 0) for t_phase in t])
    plt.plot(t, [p_swf_pos_z.evalPoly(t_phase, 1) for t_phase in t])
    plt.plot(t, [p_swf_pos_z.evalPoly(t_phase, 2) for t_phase in t])
    plt.show()