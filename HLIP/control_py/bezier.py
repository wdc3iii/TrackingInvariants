import numpy as np


class Bezier:

    def __init__(self, coeffs:np.ndarray):
        self.coeffs = coeffs
        self.order = self.coeffs.size - 1


    def eval(self, t:float) -> float:
        if self.order == 1:
            return self.evalOrder1(t)
        elif self.order == 2:
            return self.evalOrder2(t)
        elif self.order == 3:
            return self.evalOrder3(t)
        elif self.order == 4:
            return self.evalOrder4(t)
        elif self.order == 5:
            return self.evalOrder4(t)
        elif self.order == 6:
            return self.evalOrder4(t)
        else:
            raise ValueError("Order not implemented")

    def deval(self, t:float) -> float:
        if self.order == 1:
            return self.devalOrder1(t)
        elif self.order == 2:
            return self.devalOrder2(t)
        elif self.order == 3:
            return self.devalOrder3(t)
        elif self.order == 4:
            return self.devalOrder4(t)
        elif self.order == 5:
            return self.devalOrder5(t)
        elif self.order == 6:
            return self.devalOrder6(t)
        else:
            raise ValueError("Order not implemented")
    
    def ddeval(self, t:float) -> float:
        if self.order == 1:
            return self.ddevalOrder1(t)
        elif self.order == 2:
            return self.ddevalOrder2(t)
        elif self.order == 3:
            return self.ddevalOrder3(t)
        elif self.order == 4:
            return self.ddevalOrder4(t)
        elif self.order == 5:
            return self.ddevalOrder5(t)
        elif self.order == 6:
            return self.ddevalOrder6(t)
        else:
            raise ValueError("Order not implemented")

    def evalOrder1(self, t:float) -> float:
        raise ValueError("Order 1 Not Implemented Yet")

    def evalOrder2(self, t:float) -> float:
        raise ValueError("Order 2 Not Implemented Yet")

    def evalOrder3(self, t:float) -> float:
        raise ValueError("Order 3 Not Implemented Yet")

    def evalOrder4(self, t:float) -> float:
        return self.coeffs[0] * pow(1 - t, 4) + 4 * self.coeffs[1] * pow(1 - t, 3) * t + 6 * self.coeffs[2] * pow(t, 2) * pow(1 - t, 2) + 4 * self.coeffs[3] * (1 - t) * pow(t, 3) + self.coeffs[4] * pow(t, 4)

    def evalOrder5(self, t:float) -> float:
        raise ValueError("Order 5 Not Implemented Yet")

    def evalOrder6(self, t:float) -> float:
        raise ValueError("Order 6 Not Implemented Yet")
    
    def devalOrder1(self, t:float) -> float:
        raise ValueError("Order 1 Not Implemented Yet")

    def devalOrder2(self, t:float) -> float:
        raise ValueError("Order 2 Not Implemented Yet")

    def devalOrder3(self, t:float) -> float:
        raise ValueError("Order 3 Not Implemented Yet")

    def devalOrder4(self, t:float) -> float:
        return 4*(self.coeffs[1] - self.coeffs[0])*pow(1-t, 3) + 12*(self.coeffs[2] - self.coeffs[1])*pow(1-t,2)*t + 12*(self.coeffs[3]-self.coeffs[2])*(1-t)*pow(t,2) + 4*(self.coeffs[4] - self.coeffs[3])*pow(t,3)

    def devalOrder5(self, t:float) -> float:
        raise ValueError("Order 5 Not Implemented Yet")

    def devalOrder6(self, t:float) -> float:
        raise ValueError("Order 6 Not Implemented Yet")
    
    def ddevalOrder1(self, t:float) -> float:
        raise ValueError("Order 1 Not Implemented Yet")

    def ddevalOrder2(self, t:float) -> float:
        raise ValueError("Order 2 Not Implemented Yet")

    def ddevalOrder3(self, t:float) -> float:
        raise ValueError("Order 3 Not Implemented Yet")

    def ddevalOrder4(self, t:float) -> float:
        return 12*(self.coeffs[0] - 2*self.coeffs[1] + self.coeffs[2])*pow(1-t,2) + 24*(self.coeffs[1] - 2*self.coeffs[2] + self.coeffs[3])*(1-t)*t + 12*(self.coeffs[2] - 2 * self.coeffs[3] + self.coeffs[4])*pow(t, 2)

    def ddevalOrder5(self, t:float) -> float:
        raise ValueError("Order 5 Not Implemented Yet")

    def ddevalOrder6(self, t:float) -> float:
        raise ValueError("Order 6 Not Implemented Yet")
    

if __name__ == "__main__":
    # swf_x_bez = Bezier(np.array([0, 0, 0, 1, 1]))
    swf_x_bez = Bezier(np.array([0, 0.025, 0.05, 0.1, 0]))
    t = np.linspace(0, 1)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(t, [swf_x_bez.eval(tii) for tii in t])
    # plt.plot(t, [swf_x_bez.deval(tii) for tii in t])
    # plt.plot(t, [swf_x_bez.ddeval(tii) for tii in t])
    plt.show()