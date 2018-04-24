# Implement here and test.
import numpy as np


class GraphicalLasso(object):

    def __init__(self, S, convergence_threshold=0.1, lambda_param=10e-5):
        self.lambda_param = lambda_param
        self.convergence_threshold = convergence_threshold
        self.S = S

    def cov(self, S):
        # Compute the covariance of S.
        raise NotImplementedError("Implement covariance.")

    def segment(self, j, X):
        # Segment X according to (9.15) in book.
        # These should not be copies (memory views).
        raise NotImplementedError("Implement covariance.")

    def is_converged(self, W_prev, W_current):
        # Check if Frobenius norm or infinity norm of old_W - current_W is < self.convergence_threshold
        raise NotImplementedError("Implement covariance.")

    def solve(self, W_11, s_12, lambda_param):
        # Solve for beta!
        raise NotImplementedError("Implement solver.")

    def execute(self):
        W = self.cov(self.S)
        j = -1
        max_j = W.shape[0]
        while True:
            W_prev = np.copy(W)
            j = (j + 1) % max_j
            W_11, w_12, w_22 = self.segment(W, j)
            S_11, s_12, s_22 = self.segment(self.S, j)
            beta = self.solve(W_11, s_12, self.lambda_param)
            # TODO(hme): Make sure this works.
            w_12[:] = np.dot(W_11, beta)
            if self.is_converged(W_prev, W):
                break

        theta = np.empty_like(W)
        for j in range(max_j):
            W_11, w_12, w_22 = self.segment(W, j)
            S_11, s_12, s_22 = self.segment(self.S, j)
            beta = self.solve(W_11, s_12, self.lambda_param)

            theta_11, theta_12, theta_22 = self.segment(theta, j)
            theta[j, j] = 1/(w_22 - np.dot(w_12.T, beta))
            theta_12[:] = - beta * theta[j, j]
        return theta

if __name__ == "__main__":
    pass
