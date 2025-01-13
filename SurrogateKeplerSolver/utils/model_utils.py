from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
import keras
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import joblib

def non_linear_regression(x_train, y_train, model_params):

    model = make_pipeline(
        PolynomialFeatures(model_params["polynomial_degree"]),
        LinearRegression()
    )
    model.fit(x_train, y_train)

    return model

def decision_tree(X_train, y_train, model_params):
    # Create and train the multi-output gradient boosting regressor

    model = DecisionTreeRegressor(
        criterion='squared_error', max_depth=model_params["max_depth"], random_state=42)

    model.fit(X_train, y_train)

    return model


def make_decision_tree(model_params):
    # Create and train the multi-output gradient boosting regressor

    model = DecisionTreeRegressor(
        criterion='squared_error', max_depth=model_params["max_depth"], random_state=42)
    
    return model


def random_forest(X_train, y_train, model_params):
    # Create and train the multi-output gradient boosting regressor

    model = RandomForestRegressor(
        n_estimators=model_params["n_estimators"], criterion='squared_error', max_depth=model_params["max_depth"], random_state=42)

    model.fit(X_train, y_train)

    return model


def decision_tree(X_train, y_train, model_params):
    # Create and train the multi-output gradient boosting regressor

    model = DecisionTreeRegressor(criterion='squared_error', max_depth=model_params["max_depth"], random_state=42)

    model.fit(X_train, y_train)

    return model


def make_random_forest(model_params):
    # Create and train the multi-output gradient boosting regressor

    model = RandomForestRegressor(
        n_estimators=model_params["n_estimators"], criterion='squared_error', max_depth=model_params["max_depth"], random_state=42)

    return model


# def swimnetworks(X_train, y_train, x_test, y_test, model_params):

#     model = make_swimnetwork(model_params)
#     model.fit(X_train, y_train)

#     test_loss = root_mean_squared_error(y_test, model.predict(x_test))**2

#     return model, test_loss


# def make_swimnetwork(model_params):
#     steps = []

#     for i in range(model_params["n_layers"]):
#         steps.append((f"fc{i}", Dense(layer_width=model_params["layer_width"], activation=model_params[
#                      "activation"], parameter_sampler=model_params["parameter_sampler"], random_seed=42)))

#     steps.append((f"lin", Linear()))

#     model = Pipeline(steps=steps)

#     return model


def ANN(X_train, y_train, model_params):

    model = make_ANN(model_params)

    model.compile(
        optimizer=model_params["optimizer"], loss=model_params["loss_func"])

    early_stopping = keras.callbacks.EarlyStopping(monitor='loss',  # Monitor validation loss
                                                   patience=10,  # Number of epochs without improvement
                                                   restore_best_weights=True)  # Restore best weights

    model.fit(X_train, y_train, epochs=model_params["epochs"],
              batch_size=model_params["batch_size"], callbacks=model_params["callbacks"])

    return model


def make_ANN(model_params):

    model = keras.Sequential()

    # Input layer
    model.add(keras.Input(shape=(model_params["input_dims"])))

    # Hidden layers
    for i in range(model_params["n_layers"]):
        model.add(keras.layers.Dense(
            model_params["layer_width"], activation=model_params["activation"], dtype=tf.float64))
        
    # Output layer
    model.add(keras.layers.Dense(
        model_params["output_dims"], activation=model_params["activation"], dtype=tf.float64))

    return model


def save_trained_model(model, model_type, path):

    if model_type in ['RF', 'DT', 'NLR']:
        joblib.dump(model, path + ".joblib", compress=3)

    elif model_type in ['ANN']:
        model.save(path + '.keras')
