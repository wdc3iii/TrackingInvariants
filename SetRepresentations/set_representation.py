import numpy as np
import polytope as pc
from scipy.linalg import sqrtm
from abc import ABC, abstractmethod


class AbstractSet(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def sampleSet(self, N:int) -> np.ndarray:
        pass

    @abstractmethod
    def fitSet(self, points:np.ndarray) -> None:
        pass

    @abstractmethod
    def getDesc(self) -> dict:
        pass

    @abstractmethod
    def inSet(self, points:np.ndarray) -> np.ndarray:
        pass




class InftyNorm(AbstractSet):

    def __init__(self, points:np.ndarray=None, ub:np.ndarray=None, lb:np.ndarray=None) -> None:
        """Initializes an infinity norm ball (rectangular prism) of arbitrary dimension.
        Can be initialized with either a set of points to fit around, or a per-dimension upper and lower bound.

        Args:
            points (np.ndarray, optional): a set of points to fit the set around. Defaults to None.
            ub (np.ndarray, optional): the upper bound for each coordinate. Defaults to None.
            lb (np.ndarray, optional): the lower bound for each coordinate. Defaults to None.

        Raises:
            ValueError: the upper and lower bounds do not have the same shape.
        """
        super().__init__()
        if points is None and A is None and c is None:
            raise ValueError("No initialization data given. Either points or (lb, ub) must be specified.")
        if points is None and ub.shape != lb.shape:
            raise ValueError(f"Upper {ub.shape} and Lower {lb.shape} bounds not of the same shape.")

        self.ub = ub
        self.lb = lb
        self.shape = self.ub.shape[0]

        if points is not None:
            self.fitSet(points)
    
    def sampleSet(self, N: int) -> np.ndarray:
        """Samples N points uniformly from the set

        Args:
            N (int): number of points to sample from the set

        Returns:
            np.ndarray: N uniformly sampled points from the set
        """
        return np.random.uniform(self.lb, self.ub, (N, self.shape))

    def fitSet(self, points:np.ndarray) -> None:
        self.ub = np.max(points, axis=0)
        self.lb = np.min(points, axis=0)
        self.shape = self.ub.shape[0]
    
    def inSet(self, points:np.ndarray) -> np.ndarray:
        """Determine set membership of given points

        Args:
            points (np.ndarray): number of points to sample from the set

        Returns:
            np.ndarray: Set membership of given points
        """
        return np.logical_and(points <= self.ub, points >= self.lb)

    def getDesc(self) -> dict:
        return {"Method": "InftyNorm", "Desc": {"LB": np.copy(self.lb), "UB": np.copy(self.ub)}}



class Ellipsoid(AbstractSet):

    def __init__(self, points:np.ndarray=None, A:np.ndarray=None, c:np.ndarray=None, tol:float=1e-6) -> None:
        """Initializes an ellipsoid of arbitrary dimension, of the form (x-c)^T A (x-c) <= 1.
        Can be initialized with either a set of points to fit around, or values for A and c.

        Args:
            points (np.ndarray, optional): a set of points to fit the set around. Defaults to None.
            A (np.ndarray, optional): A matrix defining ellipsoid. Defaults to None.
            c (np.ndarray, optional): c vector defining ellipsoid. Defaults to None.
            tol (float, optional): the tolerance parameter when fitting a minimum volume enclosing ellipsoid. Defaults to 1e-6.

        Raises:
            ValueError: the given A and c do not have compatible dimensions.
        """
        super().__init__()
        if points is None and A is None and c is None:
            raise ValueError("No initialization data given. Either points or (A,c) must be specified.")
        if points is None and (A.shape[0] != c.shape[0] or A.shape[0] != A.shape[1]):
            raise ValueError(f"A {A.shape} and C {c.shape} do not have compatible shapes.")

        self.A = A
        self.c = c
        
        if points is not None:
            self.fitSet(points)
        
        self.d = A.shape[0]
    
    def sampleSet(self, N: int) -> np.ndarray:
        """Samples N points uniformly from the set

        Args:
            N (int): number of points to sample from the set

        Returns:
            np.ndarray: N uniformly sampled points from the set
        """
        #Sample on a unit N+1 sphere
        u = np.random.normal(0, 1, (N, self.d + 2))
        norm = np.linalg.norm(u, axis=-1,keepdims=True)
        u = u / norm
        # The first N coordinates are uniform in a unit N ball
        sampledPoints = u[:, :self.d]
        # Convert to ellispoid via transformation
        sampledPoints = (np.linalg.inv(sqrtm(self.A)) @ sampledPoints.T).T + self.c
        return sampledPoints

    def fitSet(self, points:np.ndarray) -> None:
        # Finds the ellipse equation in "center form" (x-c).T * A * (x-c) = 1
        N, d = points.shape
        Q = np.column_stack((points, np.ones(N))).T
        err = self.tol + 1.0
        u = np.ones(N) / N
        while err > self.tol:
            # assert u.sum() == 1 # invariant
            X = np.dot(np.dot(Q, np.diag(u)), Q.T)
            M = np.diag(np.dot(np.dot(Q.T, np.linalg.inv(X)), Q))
            jdx = np.argmax(M)
            step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
            new_u = (1-step_size) * u
            new_u[jdx] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u
        self.c = np.dot(u, points)
        self.A = np.linalg.inv(np.dot(np.dot(points.T, np.diag(u)), points) - np.multiply.outer(self.c, self.c)) / d
    
    def inSet(self, points:np.ndarray) -> np.ndarray:
        """Determine set membership of given points

        Args:
            points (np.ndarray): number of points to sample from the set

        Returns:
            np.ndarray: Set membership of given points
        """
        return (points - self.c).T @ self.A @ (points - self.c) <= 1

    def getDesc(self) -> dict:
        return {"Method": "Ellipsoid", "Desc": {"A": np.copy(self.A), "c": np.copy(self.c)}}


class Polytope(AbstractSet):

    def __init__(self, points=None, A=None, b=None) -> None:
        """Initializes a polytope of arbitrary dimension, of the form Ax <= b.
        Can be initialized with either a set of points to fit around, or values for A and b.

        Args:
            points (np.ndarray, optional): a set of points to fit the set around. Defaults to None.
            A (np.ndarray, optional): A matrix defining polytope. Defaults to None.
            b (np.ndarray, optional): c vector defining polytope. Defaults to None.
 
        Raises:
            ValueError: the given A and b do not have compatible dimensions.
        """
        super().__init__()
        if points is None and A is None and b is None:
            raise ValueError("No initialization data given. Either points or (A,b) must be specified.")
        if points is None and (A.shape[0] != b.shape[0] or A.shape[0] < A.shape[1]):
            raise ValueError(f"A {A.shape} and C {b.shape} do not have compatible shapes.")

        
        
        if points is not None:
            self.fitSet(points)
        else:
            self.polytope = pc.Polytope(A, b)
            self.polytope = pc.reduce(self.polytope)
            self.extreme = pc.extreme(self.polytope)
        self._findAuxPoints()
    
    def sampleSet(self, N:int) -> np.ndarray:
        boundingBox = self.polytope.bounding_box
        allSamples = np.zeros((self.polytope.dim, N))
        ii = 0
        while ii < N:
            samples = np.random.uniform(boundingBox[0], boundingBox[1], (self.polytope.dim, 4 * N - ii))
            samples = samples[:, self.inSet(samples.T)]
            if samples.shape[1] > N - ii:
                samples = samples[:, 0:N - ii]
            allSamples[:, ii:ii + samples.shape[1]] = samples
            ii += samples.shape[1]
        return allSamples.T
    
    def sampleSetHitandRun(self, N:int, thin=2) -> np.ndarray:
        """Samples N points uniformly from the set, via hit and run sampling.

        Args:
            N (int): number of points to sample from the set

        Returns:
            np.ndarray: N uniformly sampled points from the set
        """
        """Get the requested samples."""
        assert int(N) > 0
        samples = np.zeros((int(N), self.polytope.dim))
        coeff = np.random.uniform(0, 1, (self.extreme.shape[0],))
        point = coeff / np.sum(coeff) @ self.extreme
        # keep only one every thin
        for ii in range(N):
            for _ in range(thin):
                point = self._step(point)
            samples[ii, :] = point
        return samples

    def fitSet(self, points:np.ndarray) -> None:
        self.polytope = pc.qhull(points)
        self.extreme = pc.extreme(self.polytope)

    def inSet(self, points:np.ndarray) -> np.ndarray:
        """Determine set membership of given points

        Args:
            points (np.ndarray): number of points to sample from the set

        Returns:
            np.ndarray: Set membership of given points
        """
        return self.polytope.contains(points.T)

    def getDesc(self) -> dict:
        return {"Method": "Polytope", "Desc": {"A": np.copy(self.polytope.A), "b": np.copy(self.polytope)}}

    def _getRandDirection(self) -> np.ndarray:
        direction = np.random.randn(self.polytope.dim)
        return direction / np.linalg.norm(direction)
    
    def _findLambdas(self, point, direction) -> np.ndarray:
        """
        Find the lambda value for each hyperplane.

        The lambda value is the distance we have to travel
        in the current direction, from the current point, to
        reach a given hyperplane.
        """
        A = self.polytope.A
        p = self.auxPoints

        lambdas = []
        for i in range(self.polytope.A.shape[0]):
            if np.isclose(direction @ A[i], 0):
                lambdas.append(np.nan)
            else:
                lam = ((p[i] - point) @ A[i]) / (direction @ A[i])
                lambdas.append(lam)
        return np.array(lambdas)

    def _findAuxPoints(self):
        aux_points = [self._findAuxPoint(self.polytope.A[i], self.polytope.b[i])
                      for i in range(self.polytope.A.shape[0])]
        self.auxPoints = aux_points
    
    def _findAuxPoint(self, alpha, beta):
        p = np.zeros(self.polytope.dim)
        j = np.argmax(alpha != 0)
        p[j] = beta / alpha[j]
        return p
    
    def _step(self, point) -> np.ndarray:
        """Make one step."""
        # set random direction
        direction = self._getRandDirection()
        # find lambdas
        lambdas = self._findLambdas(point, direction)
        # find smallest positive and negative lambdas
        try:
            lam_plus = np.min(lambdas[lambdas > 0])
            lam_minus = np.max(lambdas[lambdas < 0])
        except(Exception):
            raise RuntimeError("The current direction does not intersect"
                               "any of the hyperplanes.")
        # throw random point between lambdas
        lam = np.random.uniform(low=lam_minus, high=lam_plus)
        # compute new point and add it
        new_point = point + lam * direction
        return new_point
    

def getExtremePoints(points:np.ndarray) -> np.ndarray:
    poly = pc.qhull(points)
    return pc.extreme(poly)

    