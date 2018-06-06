import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection, metrics, ensemble, preprocessing, datasets, pipeline, neural_network
import seaborn as sns
sns.set(style='darkgrid', context='notebook')


def lg_model():

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

    model_pipeline = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression())
    param_range = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.0, 10.0, 100.0, 1000.0]
    penalty = ['l1', 'l2']
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    param_grid = [{'logisticregression__penalty': penalty}, {'logisticregression__C': param_range},
                  {'logisticregression__solver': solver}]

    gs = model_selection.GridSearchCV(estimator=model_pipeline, param_grid=param_grid,
                                      scoring='accuracy', cv=5)

    grid_model = gs.fit(x_train, y_train)

    # Nested validation
    score = model_selection.cross_val_score(estimator=model_pipeline, X=x_train, y=y_train, cv=5, scoring='accuracy')
    print("Nested cross score: %s" % score)
    print("Nested cross average score:", np.mean(score))

    print("GridSearch best score:", grid_model.best_score_)
    print("GridSearch best params:", grid_model.best_params_)

    print("Difference (Non-nested - Nested): ", np.abs(grid_model.best_score_ - np.mean(score)))

    # Train model
    model_pipeline = grid_model.best_estimator_
    model_pipeline.fit(x_train, y_train)

    # Final score test
    prediction = model_pipeline.predict(x_test)
    print("Grid final test:", metrics.accuracy_score(y_test, prediction))

    # Ensemble learning

    log_reg = ensemble.BaggingClassifier(base_estimator=gs, n_estimators=5)
    log_reg.fit(x_train, y_train)
    prediction = log_reg.predict(x_test)
    final_score = metrics.accuracy_score(y_test, prediction)

    return final_score


def forest_model():

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

    model_pipeline = pipeline.make_pipeline(ensemble.RandomForestClassifier(n_estimators=30))
    criterion = ['gini', 'entropy']
    param_grid = [{'randomforestclassifier__criterion': criterion}]

    gs = model_selection.GridSearchCV(estimator=model_pipeline, param_grid=param_grid,
                                      scoring='accuracy', cv=5)

    grid_model = gs.fit(x_train, y_train)

    # Nested validation
    score = model_selection.cross_val_score(estimator=model_pipeline, X=x_train, y=y_train, cv=5, scoring='accuracy')
    print("Nested cross score: %s" % score)
    print("Nested cross average score:", np.mean(score))

    print("GridSearch best score:", grid_model.best_score_)
    print("GridSearch best params:", grid_model.best_params_)

    print("Difference (Non-nested - Nested): ", np.abs(grid_model.best_score_ - np.mean(score)))

    # Train model
    model_pipeline = grid_model.best_estimator_
    model_pipeline.fit(x_train, y_train)

    # Final score test
    prediction = model_pipeline.predict(x_test)
    final_score = metrics.accuracy_score(y_test, prediction)

    return final_score


def mlp_model():

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

    model_pipeline = pipeline.make_pipeline(preprocessing.StandardScaler(), neural_network.MLPClassifier())
    learning_rate_init = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2, 3, 4]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['adam', 'sgd', 'lbfgs']
    param_grid = [{'mlpclassifier__activation': activation}, {'mlpclassifier__solver': solver},
                  {'mlpclassifier__learning_rate_init': learning_rate_init}]

    gs = model_selection.GridSearchCV(estimator=model_pipeline, param_grid=param_grid,
                                      scoring='accuracy', cv=5)

    grid_model = gs.fit(x_train, y_train)

    # Nested validation
    score = model_selection.cross_val_score(estimator=model_pipeline, X=x_train, y=y_train, cv=5, scoring='accuracy')
    print("Nested cross score: %s" % score)
    print("Nested cross average score:", np.mean(score))

    print("GridSearch best score:", grid_model.best_score_)
    print("GridSearch best params:", grid_model.best_params_)

    print("Difference (Non-nested - Nested): ", np.abs(grid_model.best_score_ - np.mean(score)))

    # Train model
    model_pipeline = grid_model.best_estimator_
    model_pipeline.fit(x_train, y_train)

    # Final score test
    prediction = model_pipeline.predict(x_test)
    print("Grid final test:", metrics.accuracy_score(y_test, prediction))

    # Ensemble learning

    log_reg = ensemble.BaggingClassifier(base_estimator=gs, n_estimators=5)
    log_reg.fit(x_train, y_train)
    prediction = log_reg.predict(x_test)
    final_score = metrics.accuracy_score(y_test, prediction)

    return final_score


if __name__ == '__main__':

    # Loading data
    data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    target_names = data.target_names
    features_names = data.feature_names
    names = np.append(features_names, 'target')

    # Correlation heatmap
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    hm_size = 31

    y_ = np.reshape(y, [569, 1])
    unit_data = np.hstack((x, y_))
    cm = np.corrcoef(unit_data[0:hm_size])
    sns.set(font_scale=1)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12},
                     yticklabels=names[0:hm_size], xticklabels=names[0:hm_size])
    plt.show()
    fig.savefig('data/heatmap.png')

    # Visualization of correlation

    # !!!    NOTE: Computational expensive, VERY EXPENSIVE      !!!

    # fig = plt.figure()
    # ax = plt.axes()
    # df = pd.DataFrame(data=unit_data, columns=names)
    # ax = sns.pairplot(df[names], size=2.5)
    # fig.show()
    # fig.savefig('data/visual_correlation.png')

    # Important features
    importance_model = ensemble.GradientBoostingClassifier()
    importance_model.fit(x, y)
    importances = importance_model.feature_importances_

    fig = plt.figure()
    ax = plt.axes()

    plt.title('Feature Importances')
    ax.bar(range(x.shape[1]), importances)
    plt.xticks(range(x.shape[1]), features_names, rotation=90)
    plt.show()
    fig.savefig('data/importances.png')

    # Training and predicting

    lg_score = lg_model()        # Logistic Regression
    forest_score = forest_model()    # RandomForest
    mlp_score = mlp_model()       # MLP neural net

    print('Final Logistic Regression score: ', lg_score)
    print('Final Random Forest score ', forest_score)
    print('Final MLP score: ', mlp_score)
