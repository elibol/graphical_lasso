# Implement here and test.
import numpy as np
from sklearn.linear_model import Lasso
from scipy.linalg import sqrtm


class GraphicalLasso(object):

    def __init__(self, convergence_threshold=0.1, lambda_param=10e-5):
        """
        The Graphical Lasso algorithm implemented according to (9.1) of Statistical Learning with Sparsity.

        :param convergence_threshold: The threshold below which the Frobenius norm of the difference between the learned
                                      covariance matrix at iteration t-1 vs iteration t.
        :param lambda_param: The coefficient of the regularization term.
        """
        self.lambda_param = lambda_param
        self.convergence_threshold = convergence_threshold

    def cov(self, X):
        """
        Computes the empirical covariance given in (9.9) of Statistical Learning with Sparsity.
        """
        return np.cov(X.copy(), bias=1)

    def get_segment_indexes(self, dim, j):
        """
        Get numpy indices for segmenting matrices.
        :param dim: The dimension of a square matrix.
        :param j: The pivot index.
        :return: Numpy "fancy" indexes.
        """
        row_idx = np.concatenate((np.arange(0, j), np.arange(j+1, dim)))
        sub_mat_idx = list(map(lambda x: [x], row_idx))
        return row_idx, sub_mat_idx

    def segment(self, X, j):
        """
        Segment X according to (9.15) of Statistical Learning with Sparsity.

        :param X: The symmetric matrix to segment.
        :param j: The pivot index about which to segment X.
        :return: Segments corresponding to the sub matrix, the row matrix, and the entry X_jj.
        """
        row_idx, sub_mat_idx = self.get_segment_indexes(X.shape[0], j)
        row = X[j, row_idx]
        sub_mat = X[row_idx, sub_mat_idx]
        return sub_mat, row, X[j, j]

    def set_segment_row(self, X, j, value):
        """
        Set row entries of X according to segmentation about pivot index j.

        :param X: The matrix for which entries will be set.
        :param j: The pivot index.
        :param value: The value the rows should be set to.
        """
        row_idx, sub_mat_idx = self.get_segment_indexes(X.shape[0], j)
        X[j, row_idx] = value
        X.T[j, row_idx] = value

    def is_converged(self, W_prev, W_curr):
        """
        Check if Frobenius norm of curr_W - old_W < threshold.

        :param W_prev: The previous estimate of matrix W.
        :param W_curr: The current estimate of matrix W.
        :return: A bool indicating whether Frobenius norm of curr_W - old_W < threshold.
        """
        return (np.linalg.norm(W_curr - W_prev, 'fro') < self.convergence_threshold)

    def solve(self, W_11, s_12):
        """
        Solve for beta in (9.16) of Statistical Learning with Sparsity.

        :param W_11: The sub-matrix of W.
        :param s_12: The row vector according to the partition scheme specified in (9.15)
                     of Statistical Learning with Sparsity.
        :return: The solution to beta.
        """
        # Convert to the equations in the book (9.17, 9.18)
        Z = sqrtm(W_11)
        y = np.linalg.solve(Z, s_12)
        # Use Scikit's linear model to solve the resulting converted Lasso
        lass = Lasso(alpha=self.lambda_param, copy_X=True, fit_intercept=True, max_iter=1000,
                     normalize=False, positive=False, precompute=False, random_state=None,
                     selection='cyclic', tol=0.0001, warm_start=False)
        lass.fit(Z, y)
        return lass.coef_

    def execute(self, A):
        """
        Executes the Graphical Lasso algorithm given in (9.1) of Statistical Learning with Sparsity.

        :param A: The input matrix, with samples over the first dim and features over the second dim.
                  Entries (rows) of A are assumed to be drawn from a zero-mean multivariate Gaussian.
        :return: The estimated precision matrix of the zero-mean multivariate Gaussian distribution
                 from which samples of A are drawn.
        """
        S = self.cov(A.T)
        W = S
        j = -1
        max_j = W.shape[0]
        while True:
            W_prev = np.copy(W)
            j = (j + 1) % max_j
            W_11, w_12, w_22 = self.segment(W, j)
            S_11, s_12, s_22 = self.segment(S, j)
            beta = self.solve(W_11, s_12)
            self.set_segment_row(W, j, np.dot(W_11, beta))
            if self.is_converged(W_prev, W):
                break

        theta = np.empty_like(W)
        for j in range(max_j):
            W_11, w_12, w_22 = self.segment(W, j)
            S_11, s_12, s_22 = self.segment(S, j)
            beta = self.solve(W_11, s_12)

            theta[j, j] = 1/(w_22 - np.dot(w_12.T, beta))
            self.set_segment_row(theta, j, - beta * theta[j, j])
        return theta


def is_psd(X):
    return np.all(np.linalg.eigvals(X) >= 0)


if __name__ == "__main__":
    # Basic tests.
    lasso = GraphicalLasso()

    # Test covariance.
    X = np.array([[1, 3, 5, 7],
                  [11, 13, 17, 23]])
    S = lasso.cov(X.T)
    # It's symmetric.
    assert np.allclose(S, S.T)
    # It has the right dimensions
    assert S.shape[0] == X.shape[1]

    # Test segmentation.
    Y = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])
    for j in range(Y.shape[0]):
        Y_11, Y_12, Y_22 = lasso.segment(Y, j)
        print("j", j)
        print("Y_11", Y_11)
        print("Y_12", Y_12)
        print("Y_22", Y_22)

    # Test setting of rows.
    print(Y)
    for j in range(Y.shape[0]):
        value = np.array([10+j, 10+j])
        lasso.set_segment_row(Y, j, value)
        Y_11, Y_12, Y_22 = lasso.segment(Y, j)
        assert np.allclose(Y_12, value)
        assert np.allclose(Y_12.T, value)
        print(Y)

    # Identical matrices considered converged.
    assert lasso.is_converged(Y, Y.T)

    # TODO(barry): Simple test for the solver.

    # Test the execute code path (not testing for correctness).
    A = np.array([[1, -1],
                  [-1, 1],
                  [1, 1],
                  [1, -1]])
    precision_mat = lasso.execute(A)
    print("precision_mat: ", precision_mat)
