__author__ = 'Shanmukh'
import sys
import numpy as np
import codecs
from numpy import linalg as LA

def build_movies_dict(movies_file):
    movie_id_dict = {}
    with codecs.open(movies_file, 'r', 'latin-1') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            movieId, title, genres = line.split(',')
            movie_id_dict[int(movieId)] = i - 1
    return movie_id_dict

def read_data(input_file, movies_dict):
    users = 718
    movies = 8927
    X = np.zeros(shape=(users, movies))
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            user, movie_id, rating, timestamp = line.split(',')
            id = movies_dict[int(movie_id)]
            X[int(user) - 1, id] = float(rating)
    return X

def matrix_factorization(X, P, Q, K, steps, alpha, beta):
    Q = Q.T
    for step in range(steps):
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i][j] > 0:
                    eij = X[i][j] - np.dot(P[i, :], Q[:, j])
                    regularization_term = beta * (LA.norm(P) + LA.norm(Q))
                    eij += regularization_term
                    for k in range(K):
                        P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        error = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i][j] > 0:
                    error += np.power(X[i][j] - np.dot(P[i, :], Q[:, j]), 2)
        if error < 0.001:
            break
    return P, Q.T

def main(X, K):
    N = X.shape[0]
    M = X.shape[1]
    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)
    steps = 5000
    alpha = 0.0002
    beta = 0.02
    estimated_P, estimated_Q = matrix_factorization(X, P, Q, K, steps, alpha, beta)
    modeled_X = np.dot(estimated_P, estimated_Q.T)
    np.savetxt('mf_result.txt', modeled_X, delimiter=',')

if __name__ == '__main__':
    if len(sys.argv) == 4:
        ratings_file = sys.argv[1]
        no_of_features = int(sys.argv[2])
        movies_mapping_file = sys.argv[3]
        movies_dict = build_movies_dict(movies_mapping_file)
        numpy_arr = read_data(ratings_file, movies_dict)
        main(numpy_arr, no_of_features)
