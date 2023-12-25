__author__ = 'Shanmukh'
import sys
import numpy as np
import codecs
import operator

def dict_with_movie_and_id(movies_file):
    movies_names_dict = {}
    movies_id_dict = {}
    i = 0
    with codecs.open(movies_file, 'r', 'latin-1') as f:
        for line in f:
            if i == 0:
                i += 1
            else:
                movie_id, movie_name, genre = line.split(',')
                movies_names_dict[int(movie_id)] = movie_name
                movies_id_dict[int(movie_id)] = i - 1
                i += 1
    return movies_names_dict, movies_id_dict

def dict_with_user_unrated_movies(rating_file, movie_mapping_id):
    users = 718
    movies = 8927
    dict_with_unrated_movies_users = {}
    X = np.zeros(shape=(users, movies))
    with open(rating_file, 'r') as f:
        for line in f:
            user, movie, rating, timestamp = line.split(',')
            id = movie_mapping_id[int(movie)]
            X[int(user) - 1, id] = float(rating)
    for row in range(X.shape[0]):
        unrated_movie_ids = np.nonzero(X[row] == 0)
        unrated_movie_ids = list(map(lambda x: x + 1, unrated_movie_ids[0]))
        dict_with_unrated_movies_users[row + 1] = unrated_movie_ids
    return dict_with_unrated_movies_users

def build_predicted_numpy_array(pred_file):
    users = 718
    movies = 8927
    X = np.zeros(shape=(users, movies))
    user = 0
    with open(pred_file, 'r') as f:
        for line in f:
            ratings = line.split(',')
            for movie_id, rating in enumerate(ratings):
                X[user, movie_id] = float(rating)
            user += 1
    return X

def top_25_recommended_movies(pred_rating_file, users, unrated_movies_per_user, movies_mapping_names, movie_mapping_id):
    reverse_movie_id_mapping = {val: key for key, val in movie_mapping_id.items()}
    for user in users:
        dict_pred_unrated_movies = {}
        unrated_movies = unrated_movies_per_user[int(user)]
        for unrated_movie in unrated_movies:
            dict_pred_unrated_movies[int(unrated_movie)] = pred_rating_file[int(user) - 1][int(unrated_movie) - 1]
        sorted_movies = sorted(dict_pred_unrated_movies.items(), key=operator.itemgetter(1), reverse=True)
        print(f"Top 25 movies recommendation for the user {user}")
        for i in range(25):
            movie_id, rating = sorted_movies[i]
            if rating >= 3.5:
                print(f"{movies_mapping_names[reverse_movie_id_mapping[movie_id]]}")
        print("\n")

def recommend_movies_for_users(orig_rating_file, pred_rating_file, movies_file, users):
    movies_mapping_names, movie_mapping_id = dict_with_movie_and_id(movies_file)
    predicted_rating_numpy_array = build_predicted_numpy_array(pred_rating_file)
    dict_with_unrated_movies_users = dict_with_user_unrated_movies(orig_rating_file, movie_mapping_id)
    top_25_recommended_movies(predicted_rating_numpy_array, users, dict_with_unrated_movies_users, movies_mapping_names, movie_mapping_id)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        orig_rating_file = sys.argv[1]
        pred_rating_file = sys.argv[2]
        movies_file = sys.argv[3]
        list_of_users = sys.argv[4]
        with open(list_of_users, 'r') as f:
            users = f.readline().split(',')
        recommend_movies_for_users(orig_rating_file, pred_rating_file, movies_file, users)
