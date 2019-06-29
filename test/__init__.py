import numpy as np
from src.quadprobb import quadprogbb, cvxlpbnd


def test_quadprodbb():
    Q = np.array([[19.42941344, -12.9899322, -5.1907171, -0.25782677],
                  [-12.9899322, 15.97805787, 1.87087712, -6.72150886],
                  [-5.1907171, 1.87087712, 36.99333345, -16.21139038],
                  [-0.25782677, -6.72150886, -16.21139038, 103.0762929]])

    q = np.array([-45.3553788, 26.52058282, -99.63769322, -61.59361441])

    mean_lower = np.array([4.94791667, 3.36875, 1.41666667, 0.19375])
    mean_upper = np.array([5.04375, 3.46458333, 1.5125, 0.28958333])

    solution = quadprogbb(Q, q, LB=mean_lower, UB=mean_upper)
    print(solution)


test_quadprodbb()


