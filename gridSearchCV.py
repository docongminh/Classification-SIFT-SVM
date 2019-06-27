from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, cross_val_score
from sklearn.svm import SVC


# X_train, Y_train, X_test, Y_test from build_model.py

pipe_svc = Pipeline(
    ['scl', StandardScaler()],
    ['clf', SVC(random_state = 0, probability = True)]
)

param_c = [5, 10, 15, 20, 25, 30, 35, 40]
param_gamma = [0.01, 0.02, 0.05, 0.1, 0.2]

param_grid = [
    {
        'clf__C': param_c,
        'clf__kernel': ['linear']
    },

    {
        'clf__C': param_c,
        'clf__gamma': param_gamma,
        'clf__kernel': ['rbf']

    }

]

gs_svm = GridSearchCV(
    estimator = pipe_svc,
    param_grid = param_grid,
    scoring = 'accuracy',
    cv = 5
)

# training

gs_svm.fit(X_train, Y_train)

print("Optimal SVm parameters: ", gs_svm.best_params_)
print("Score with best params: ", gs_svm.score(X_test, Y_test))
score = cross_val_score(gs_svm, X_train, X_test, scoring = 'accuracy, cv = 5')
print("Average accuracy: %.4f +/- %.4f" % ((np.mean(score)), np.std(score)))





