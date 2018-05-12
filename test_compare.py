import graphical_lasso as gl
import preprocessing as pp
import numpy as np
from sklearn import preprocessing


def get_other_precision(A):
    # reference on sklearn's graph lasso: http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphLasso.html
    from sklearn.covariance import GraphLasso  # our Algo code should replace this and input/output the same thing

    graph_lasso = GraphLasso(
        alpha=1e-5)  # alpha =  regularization parameter: the higher alpha, the more regularization, the sparser the inverse covariance.
    graph_lasso.fit(A)  # A is the aggregated sentiment matrix, an arrray of (n_samples, n_features)
    precision = graph_lasso.get_precision()
    return precision


def get_precision(A):
    # divide by 21 since this is what sk-learn does internally.
    lasso = gl.GraphicalLasso(convergence_threshold=1e-6, lambda_param=1e-5/21)
    precision = lasso.execute(A)
    return precision


def main(topic="isis"):
    # get A
    A, sources = pp.get_A_and_labels(topic)
    A = A.astype('float64')
    if topic == "brexit":
        # Add noise to brexit.
        A += np.random.randn(A.shape[0], A.shape[1])*1e-16

    precision_1 = get_other_precision(A)
    # precision_1 = get_precision(A + np.random.randn(A.shape[0], A.shape[1])*1e-16)
    precision_2 = get_precision(A)

    print(np.allclose(precision_1, precision_2))
    print(np.max(np.abs(precision_1-precision_2)))

    # print pairs of sources for which precision matrix has pos val
    # when precision matrix is pos, source pairs are likely to have same sentiment
    # when precision matrix is neg, source pairs are likely to have opposit sentiment
    a = []
    b = []
    for (i, j) in zip(*np.where(precision_1 > 0)):
        if i > j: #since precision matrix is symmetric, only need to print upper half
            a.append((set([sources[i], sources[j]]), precision_1[i][j]))

    for (i, j) in zip(*np.where(precision_2 > 0)):
        if i > j: #since precision matrix is symmetric, only need to print upper half
            b.append((set([sources[i], sources[j]]), precision_2[i][j]))

    a = sorted(a)
    b = sorted(b)

    # print("\nPrecision (sklearn)")
    # print("\nPrecision (us)")
    for i in range(min(len(a), len(b))):
        print(a[i], b[i])

if __name__ == "__main__":
    main("isis")
    # main("brexit")
