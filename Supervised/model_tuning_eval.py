import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sklearn import linear_model, model_selection, metrics, svm, tree, ensemble, preprocessing, datasets, pipeline
from Supervised import utils
from mlxtend.plotting import plot_decision_regions
import seaborn as sns; sns.set()


def my_learning_curve(model, x_train, y_train):
    """

       Via the train_sizes parameter in the learning_curve function, we can control the
    absolute or relative number of training samples that are used to generate the learning
    curves. Here, we set train_sizes=np.linspace(0.1, 1.0, 10) to use 10 evenly
    spaced relative intervals for the training set sizes. By default, the learning_curve
    function uses stratified k-fold cross-validation to calculate the cross-validation
    accuracy, and we set k =10 via the cv parameter. Then, we simply calculate the
    average accuracies from the returned cross-validated training and test scores for the
    different sizes of the training set, which we plotted using matplotlib's plot function.
    Furthermore, we add the standard deviation of the average accuracies to the plot
    using the fill_between function to indicate the variance of the estimate.

    """
    train_sizes, train_scores, test_scores = model_selection.learning_curve(estimator=model,
                                                                            X=x_train, y=y_train,
                                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                                            cv=10, n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')
    plt.fill_between(train_sizes,train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean,  color='green', linestyle='--',
             marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.title('Learning curve')
    plt.show()


def my_validation_curve(model, X_train, Y_train):

    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    train_scores, test_scores = model_selection.validation_curve(estimator=model, X=X_train,
                                                                 y=Y_train, param_name='svc__C',
                                                                 param_range=param_range, cv=10)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, color='green', linestyle='--',
             marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    plt.title("Validation curve")
    plt.show()


def confusion_matrix(data, y_test, prediction):
    conf_mat = metrics.confusion_matrix(y_test, prediction)
    print(conf_mat)

    sns.heatmap(conf_mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=data.target_names, yticklabels=data.target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.title('Confusion matrix')
    plt.show()


def main():
    data = datasets.load_iris()
    x = data.data
    y = data.target

    # x, y = datasets.load_iris(return_X_y=True)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

    # GridSearch
    model_pipeline = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.SVC())
    param_range = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
                  {'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}]

    gs = model_selection.GridSearchCV(estimator=model_pipeline, param_grid=param_grid,
                                      scoring='accuracy', cv=3)

    gs = gs.fit(x_train, y_train)

    # Nested validation
    score = model_selection.cross_val_score(estimator=model_pipeline, X=x_train, y=y_train, cv=5, scoring='accuracy')
    print("Nested cross score: %s" % score)
    print("Nested cross average score:", np.mean(score))

    print("GridSearch best score:", gs.best_score_)
    print("GridSearch best params:", gs.best_params_)

    print("Difference (Non-nested - Nested): ", np.abs(gs.best_score_ - np.mean(score)))

    # Train model
    model_pipeline = gs.best_estimator_
    model_pipeline.fit(x_train, y_train)

    # Final score test
    prediction = model_pipeline.predict(x_test)
    print("Final test:", metrics.accuracy_score(y_test, prediction))

    # Using validation and learning curves to evaluate parameters
    my_learning_curve(model_pipeline, x_train, y_train)
    my_validation_curve(model_pipeline, x_train, y_train)

    # Using metrics
    confusion_matrix(data, y_test, prediction)


if __name__ == '__main__':
    main()