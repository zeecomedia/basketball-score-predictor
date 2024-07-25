from sklearn.ensemble import GradientBoostingRegressor

def find_ci(X_train, X_test, y_train):
    # Set lower and upper quantile
    LOWER_ALPHA = 0.2
    UPPER_ALPHA = 0.8

    lower_model = GradientBoostingRegressor(loss="quantile",
                                            alpha=LOWER_ALPHA)
    upper_model = GradientBoostingRegressor(loss="quantile",
                                            alpha=UPPER_ALPHA)

    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)

    return lower_model.predict(X_test), upper_model.predict(X_test)

def return_arrays(lower, upper):
    list = []
    for i in range(len(lower)):
        list.append((int(lower[i].round()), int(upper[i].round())))
    return list