import itertools
import json
import keras
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sympy import hyper

from utils import model_utils


def do_hyperparam_search(model_name, dataset, hyperparams, save_path, extra_title = ""):
    
    extension = ".json"

    if model_name == 'RF':
        results = do_hyperparam_search_RF(dataset, hyperparams)

    elif model_name == 'ANN':
        results = do_hyperparam_search_ANN(dataset, hyperparams)

    elif model_name == 'swim':
        results = do_hyperparam_search_swim(dataset, hyperparams)
    
    elif model_name == 'DT':
        results = do_hyperparam_search_DT(dataset, hyperparams)
        
    save_path = save_path + "_" + model_name + extra_title + extension
    
    save_hyp_search_results(results, save_path)



def split_dataset(dataset, hyperparams):

    x_train, x_val, y_train, y_val = train_test_split(dataset[:, :hyperparams["input_dims"][0]], dataset[:, hyperparams["input_dims"][0]:], test_size=hyperparams["test_size"], random_state=42)

    return x_train, x_val, y_train, y_val


def do_hyperparam_search_swim(dataset, hyperparams):

    x_train, x_val, y_train, y_val = split_dataset(dataset, hyperparams)

    # Prepare search space
    layer_width_options = hyperparams.get('layer_widths')
    num_layers_options = hyperparams.get('num_layers')
    
    # Results storage
    results = []
    
    # Hyperparameter search (grid search)
    for num_layers in num_layers_options:
        for base_width in layer_width_options:
            
            # for layer_config in layer_configs:
                # Create and train model
            hyperparams["n_layers"] = num_layers
            hyperparams["layer_width"] = base_width
            model = model_utils.make_swimnetwork(hyperparams)
            
            print(f"Fitting Swim with {num_layers} layers and {base_width} layer width")
            # Fit model
            model.fit(x_train, y_train)
            
            # Evaluate model
            val_loss = mean_squared_error(model.predict(x_val), y_val)
            
            # Store results
            results.append({
                'layer_widths': base_width,
                'num_layers': num_layers,
                'val_loss': val_loss
            })
    
    # Sort results by validation accuracy
    results.sort(key=lambda x: x['val_loss'], reverse=False)
    
    return results

def do_hyperparam_search_RF(dataset, hyperparams):

    x_train, x_val, y_train, y_val = split_dataset(dataset, hyperparams)

    # Prepare search space
    n_estimators_options = hyperparams.get('n_estimators_options')
    max_depth_options = hyperparams.get('max_depth_options')
    
    # Results storage
    results = []
    
    # Hyperparameter search (grid search)
    for n_estimators in n_estimators_options:
        for max_depth in max_depth_options:
            
            # for layer_config in layer_configs:
            # Create and train model
            hyperparams["n_estimators"] = n_estimators
            hyperparams["max_depth"] = max_depth
            model = model_utils.make_random_forest(hyperparams)
            
            print(f"Fitting RF with {n_estimators} estimators and {max_depth} max depth")

            # Fit model
            model.fit(x_train, y_train)
            
            # Predict on validation set
            val_pred = model.predict(x_val)
            train_pred = model.predict(x_train)
            
            # Calculate metrics
            val_loss = hyperparams['loss_func'](y_val, val_pred)
            train_loss = hyperparams['loss_func'](train_pred, y_train)
            
            # Store results
            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'val_loss': val_loss,
                'train_loss': train_loss
            })
    
    # Sort results by validation loss
    results.sort(key=lambda x: x['val_loss'], reverse=False)
    
    return results

def do_hyperparam_search_DT(dataset, hyperparams):

    x_train, x_val, y_train, y_val = split_dataset(dataset, hyperparams)

    # Prepare search space
    max_depth_options = hyperparams.get('max_depth_options')
    
    # Results storage
    results = []
    
    for max_depth in max_depth_options:
        
        # for layer_config in layer_configs:
        # Create and train model
        hyperparams["max_depth"] = max_depth
        model = model_utils.make_decision_tree(hyperparams)
        
        print(f"Fitting DT with {max_depth} max depth")

        # Fit model
        model.fit(x_train, y_train)
        
        # Predict on validation set
        val_pred = model.predict(x_val)
        train_pred = model.predict(x_train)
        
        # Calculate metrics
        val_loss = hyperparams['loss_func'](y_val, val_pred)
        train_loss = hyperparams['loss_func'](train_pred, y_train)
        
        # Store results
        results.append({
            'max_depth': max_depth,
            'val_loss': val_loss,
            'train_loss': train_loss
        })
    
    # Sort results by validation loss
    results.sort(key=lambda x: x['val_loss'], reverse=False)
    
    return results


def do_hyperparam_search_ANN(dataset, hyperparams):

    x_train, x_val, y_train, y_val = split_dataset(dataset, hyperparams)

    # Prepare search space
    layer_width_options = hyperparams.get('layer_widths')
    num_layers_options = hyperparams.get('num_layers')
    
    # Results storage
    results = []
    
    # Hyperparameter search (grid search)
    for num_layers in num_layers_options:
        for base_width in layer_width_options:
            # Generate layer width combinations
            # layer_configs = list(itertools.combinations_with_replacement(base_width, num_layers))
            
            # for layer_config in layer_configs:
                # Create and train model
            hyperparams["n_layers"] = num_layers
            hyperparams["layer_width"] = base_width
            model = model_utils.make_ANN(hyperparams)

            model.compile(optimizer=hyperparams["optimizer"], loss=hyperparams["loss_func"], metrics=['accuracy'])
            
            # Early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            )
            
            print(f"Fitting ANN with {num_layers} layers and {base_width} layer width")
            # Fit model
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=hyperparams["epochs"],
                batch_size=hyperparams["batch_size"],
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate model
            val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
            
            # Store results
            results.append({
                'layer_widths': base_width,
                'num_layers': num_layers,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'training_history': history.history
            })
    
    # Sort results by validation accuracy
    results.sort(key=lambda x: x['val_accuracy'], reverse=True)
    
    return results

def save_hyp_search_results(results, save_path):

    with open(save_path, 'w') as f:
        json.dump(results, f)