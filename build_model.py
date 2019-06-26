from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn
from data_loader import label_image
from sift_extractors import create_feature_bow, extract_sift_features, kmean_bow
import pickle
import argparse
import os

arg = argparse.ArgumentParser()
arg.add_argument("-dt", "--inputdata", required = True, help = "path to data training")
args = vars(arg.parse_args())

path = args["inputdata"]
print(path)
data_train, label, label2id = label_image(path)

image_desctiptors = extract_sift_features(data_train)

all_descriptors = []
for descriptor in image_desctiptors:
    if descriptor is not None:
        for des in descriptor:
            all_descriptors.append(des)

num_cluster = 60
BoW = kmean_bow(all_descriptors, num_cluster)
# if os.path.isfile('bow_dictionary.pkl'):
#     BoW = kmean_bow(all_descriptors, num_cluster)
# else:
#     BoW = pickle.load(open('bow_dictionary.pkl', 'rb'))

X_features = create_feature_bow(image_desctiptors, BoW, num_cluster)

X_train, X_test, Y_train, Y_test = train_test_split(X_features, label, test_size = 0.2, random_state = 1)
model_svm = sklearn.svm.SVC(C = 30, random_state = 0)
# parameters = [
#     {'C': [20, 25, 30, 35, 40, 45]}

# ]
# grid_model = GridSearchCV(
#     estimator = model_svm,
#     parameters = parameters,
#     cv = 10
#
# )
# grid_model.fit(X_train, Y_train)

model_svm.fit(X_train, Y_train)
filename = 'svm_model.sav'
pickle.dump(model_svm, open(filename, 'wb'))
print("score on training set params: ", model_svm.score(X_train, Y_train))
print("score on testing set params: ", model_svm.score(X_test, Y_test))
# print("best score: ", grid_model.best_score_)
# print("best_params: ", grid_model.best_params_)
