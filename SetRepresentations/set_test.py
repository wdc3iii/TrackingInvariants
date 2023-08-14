import time
import numpy as np
from SetRepresentations.set_representation import InftyNorm, Ellipsoid, Polytope, getExtremePoints

N = 20
def main():
    pts = np.random.uniform(0, 1, (N, 3))

    t0 = time.time()
    poly = Polytope(points=pts)
    t1 = time.time()
    ext = getExtremePoints(pts)
    t2 = time.time()
    samples = poly.sampleSet(1000)
    t3 = time.time()
    # samples = poly.sampleSet2(1000)
    # t4 = time.time()

    print(f"Constructing Polytope: {t1 - t0}\nGetting Extreme points: {t2 - t1}\nSampling set: {t3 - t2}\n")
    # print(f"Constructing Polytope: {t1 - t0}\nGetting Extreme points: {t2 - t1}\nSampling set: {t3 - t2}\nRejection: {t4 - t3}")

    A = np.array([
    [1.0000, 0, 0],
    [-1.0000, 0, 0],
    [0, 0,    1.0000],
    [0, 0,   -1.0000],
    [-1.8305,   -0.3792,    1.0000],
    [3.2654,    0.8076,   -0.9708],
    [-3.2654,   -0.8076,    0.9708]
    ])
    b = np.array([
        0.0750,
    0.0750,
    0.1000,
         0,
    0.0750,
    0.0411,
    0.0000
    ])

    t0 = time.time()
    poly2 = Polytope(A=A, b=b)
    t1 = time.time()
    samples = poly2.sampleSet(1000)
    t2 = time.time()
    print(f"\nConstructing Polytope: {t1 - t0}\nSampling Set: {t2 - t1}\n")


if __name__ == "__main__":
    main()