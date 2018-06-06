import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model, model_selection, metrics, ensemble, preprocessing, datasets, pipeline
import seaborn as sns
sns.set(style='darkgrid', context='notebook')


def traditional_regression():

    # Visualization of correlation
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV(target)']
    sns.pairplot(df[cols], size=2.5)
    plt.show()


    # Correlation heatmap
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols, xticklabels=cols)
    plt.show()


    # Creating simple model
    lr = pipeline.make_pipeline(linear_model.LinearRegression())
    lr.fit(x_train, y_train)
    print('SimpleModel\n', 'Prediction:', lr.predict([x_test[0, :]]), 'True value:', y_test[0])


    # Creating grid model
    my_pipeline = pipeline.make_pipeline(preprocessing.PolynomialFeatures(), linear_model.LinearRegression())
    features_params = [1, 2, 3]
    param_grid = [{'polynomialfeatures__degree': features_params}]
    gs = model_selection.GridSearchCV(estimator=my_pipeline, param_grid=param_grid)

    gs = gs.fit(x_train, y_train)
    model_pipeline = gs.fit(x_train, y_train).best_estimator_

    print('\nGridModel\n', 'Prediction:', model_pipeline.predict([x_test[0, :]]), 'True value:', y_test[0])
    print(gs.best_params_)


    # Evaluation
    y_train_pred = model_pipeline.predict(x_train)
    y_test_pred = model_pipeline.predict(x_test)

    plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='o', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
    plt.xlim([-10, 50])
    plt.show()

    print('\nEVALUATION\n')
    print('\nMSE train: %.3f, test: %.3f' % (metrics.mean_squared_error(y_train, y_train_pred),
                                             metrics.mean_squared_error(y_test, y_test_pred)))
    print('\nR^2 train: %.3f, test: %.3f' % (metrics.r2_score(y_train, y_train_pred),
                                             metrics.r2_score(y_test, y_test_pred)))


def tranforming_features():
    # Transforming the dataset:

    X = df[['LSTAT']].values
    y = df['MEDV(target)'].values

    regr = linear_model.LinearRegression()

    # create quadratic features
    quadratic = preprocessing.PolynomialFeatures(degree=2)
    cubic = preprocessing.PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    # fit features
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

    regr = regr.fit(X, y)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = metrics.r2_score(y, regr.predict(X))

    regr = regr.fit(X_quad, y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = metrics.r2_score(y, regr.predict(X_quad))

    regr = regr.fit(X_cubic, y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    cubic_r2 = metrics.r2_score(y, regr.predict(X_cubic))


    # plot results
    plt.scatter(X, y, label='training points', color='lightgray')

    plt.plot(X_fit, y_lin_fit,
             label='linear (d=1), $R^2=%.2f$' % linear_r2,
             color='blue',
             lw=2,
             linestyle=':')

    plt.plot(X_fit, y_quad_fit,
             label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
             color='red',
             lw=2,
             linestyle='-')

    plt.plot(X_fit, y_cubic_fit,
             label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
             color='green',
             lw=2,
             linestyle='--')

    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('Price in $1000s [MEDV(target)]')
    plt.legend(loc='upper right')

    #plt.savefig('images/10_11.png', dpi=300)
    plt.show()


    X = df[['LSTAT']].values
    y = df['MEDV(target)'].values

    # transform features
    X_log = np.log(X)
    y_sqrt = np.sqrt(y)

    # fit features
    X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]

    regr = regr.fit(X_log, y_sqrt)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = metrics.r2_score(y_sqrt, regr.predict(X_log))

    # plot results
    plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')

    plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2)

    plt.xlabel('log(% lower status of the population [LSTAT])')
    plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV(target)]}$')
    plt.legend(loc='lower left')

    plt.tight_layout()
    #plt.savefig('images/10_12.png', dpi=300)
    plt.show()







# # Dealing with nonlinear relationships using random forests

# ...

# ## Decision tree regression



def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return


def decision_tree_regression():
    X = df[['LSTAT']].values
    y = df['MEDV(target)'].values

    tree = tree.DecisionTreeRegressor(max_depth=3)
    tree.fit(X, y)

    sort_idx = X.flatten().argsort()

    lin_regplot(X[sort_idx], y[sort_idx], tree)
    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('Price in $1000s [MEDV(target)]')
    #plt.savefig('images/10_13.png', dpi=300)
    plt.show()



    # ## Random forest regression



    X = df.iloc[:, :-1].values
    y = df['MEDV(target)'].values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.4, random_state=1)





    forest = ensemble.RandomForestRegressor(n_estimators=1000,
                                   criterion='mse',
                                   random_state=1,
                                   n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    print('MSE train: %.3f, test: %.3f' % (
            metrics.mean_squared_error(y_train, y_train_pred),
            metrics.mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            metrics.r2_score(y_train, y_train_pred),
            metrics.r2_score(y_test, y_test_pred)))


    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', edgecolor='white',
                marker='o', s=35, alpha=0.9, label='training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', edgecolor='white',
                marker='s', s=35, alpha=0.9, label='test data')

    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
    plt.xlim([-10, 50])
    plt.tight_layout()

    # plt.savefig('images/10_14.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    data = datasets.load_boston()
    x, y = datasets.load_boston(return_X_y=True)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

    d = {'CRIM': data.data[:, 0], 'ZN': data.data[:, 1], 'INDUS': data.data[:, 2], 'CHAS': data.data[:, 3],
         'NOX': data.data[:, 4], 'RM': data.data[:, 5], 'AGE': data.data[:, 6], 'DIS': data.data[:, 7],
         'RAD': data.data[:, 8], 'TAX': data.data[:, 9], 'PTRATIO': data.data[:, 10],
         'B': data.data[:, 11], 'LSTAT': data.data[:, 12], 'MEDV(target)': data.target}
    df = pd.DataFrame(data=d)

    traditional_regression()
    tranforming_features()
    decision_tree_regression()