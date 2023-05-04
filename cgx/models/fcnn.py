"""
Wrapper function to train generic tensorflow DNNs to be explained
"""
import tensorflow as tf
from typing import *
from tensorflow.keras import backend as K
import keras_tuner as kt
import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
from tensorflow.keras.utils import to_categorical
from functools import partial
import logging
from pathlib import Path
from cgx.models.utils import load_tuner, save_tuner
import os
from box import Box


################################################################################
## Training loop
################################################################################
def train_dnn(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    fold: int = 0,
    objective: str = "val_auc",
    objective_dir: str = "max",
    tuner_epochs: int = 20,
    tuner_factor: int = 3,
    tuner_overwrite: bool = True,
    final_callbacks: Union[List, None] = None,
    ):
    """
    Wrapper which conducts hyperparameter search of the DNN and serialises the model

    Args:
        dataset (str): name of the dataset (used in model filepath)
        fold (int): experiment fold used as random seed
        save_path (str): base path to save the model
    """

    # define model topology/function that the tuner should optimise
    model_fn_wrapper = partial(model_fn,
                               input_features=x_train.shape[1],
                               num_outputs=y_train.nunique(),
                               layers=[64,32,16])

    # run hyperparameter tuning only for one of the folds
    unique_elems = np.unique(y_train)
    class_weights = dict(enumerate(
        compute_class_weight(
            'balanced',
            unique_elems,
            y_train
        )
    ))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=objective, patience=5)

    # tuner_path = Path(config.dnn_path).joinpath(f"{dataset}/tuner.pkl")

    # if fold == 0 or not use_pickle or not os.path.exists(tuner_path):
        # compute class weights
    tuner = kt.Hyperband(hypermodel=model_fn_wrapper,
                         objective=kt.Objective(objective, direction=objective_dir),
                         max_epochs=tuner_epochs,
                         factor=tuner_factor,
                         seed=fold,
                         overwrite=tuner_overwrite)

    tuner.search(x_train, to_categorical(y_train), epochs=20, validation_split=0.1, callbacks=[early_stopping], class_weight = class_weights, batch_size=16)
    #     save_tuner(tuner, config, dataset)
    # else:
    #     tuner = load_tuner(config, dataset)

    # grab best hyperparameters from the search
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    logging.info(f"Best hyperparameters: {best_hps.values}")

    model = tuner.hypermodel.build(best_hps)
    y_train_cat = to_categorical(y_train)
    # fit best model
    history = model.fit(x_train, y_train_cat, validation_split=0.1, verbose=1, epochs=40, class_weight=class_weights, batch_size=16, callbacks=final_callbacks)

    # evaluate model
    loss, logit_auc, auc, _, accuracy, f1 = model.evaluate(x_test, to_categorical(y_test), verbose=1)
    metrics = {
        "loss": loss,
        "auc": auc,
        "accuracy": accuracy,
        "f1": f1
    }

    logging.info(f"Model test performance: {metrics}")

    return model, metrics



################################################################################
## Model Definition
################################################################################
def model_fn(
    hp,
    input_features: int,
    num_outputs: int,
    layers: List[int],
):
    """
    Main function to build and compile the model with hyperparameter tuning using the Keras ``HyperTuner``
    enabled. Note that all the ``hp_<name>`` variables refer to variables that are tried during the
    hyperparameter search.

    Args:
        hp (keras_tuner.engine.hyperparameters.HyperParameters): keras tuner hyperparameter to pass into
            the HyperTuner. Since the function takes additional arguments, the only way to pass this into
            the HypterTuner is by wrapping the function call in `functools.partial` and specifying the
            additional arguments there

            Example
            ```
            model_fn_wrapper = partial(model_fn, input_features=x_train.shape[1], num_outputs=y_train.nunique(), layers=[64, 32, 16])
            tuner = kt.Hyperband(hypermodel=model_fn_wrapper, ...)
            ```
        input_features (int): dimensions of training data for input layer (number of features)
        layers (List[int]): Sequential specification of the Dense layer dimensions. For example, passing
            ```[64, 32, 16]`` will result in the first dense layer having 64 nodes

    Returns:
        tf.model: Compiled tensorflow model
    """



    # Input layer
    input_layer = tf.keras.layers.Input(shape=(input_features,))

    # define hyperparameter search space
    hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    hp_activation = hp.Choice(name="hidden_layer_activation", values=["tanh", "relu"])
    hp_l2_reg = hp.Choice(name="l2_regularization", values=[0.001, 0.002, 0.005])
    hp_hidden_reg = hp.Choice(name="regularise_hidden_layers", values=[True, False])
    hp_norm = hp.Choice(name="batch_norm", values=[True, False])

    hp_dropout = hp.Choice("dropout", values=[0.0, 0.2])


    normalizer = tf.keras.layers.BatchNormalization(axis=-1)


    model = tf.keras.Sequential()
    model.add(input_layer)
    # add batch normalisation
    if hp_norm:
        model.add(normalizer)


    for i, units in enumerate(layers):
        model.add(tf.keras.layers.Dense(
            units=units,
            activation=hp_activation,
            kernel_regularizer=tf.keras.regularizers.l2(hp_l2_reg)
        ))
        model.add(tf.keras.layers.Dropout(hp_dropout))


    # add output layer
    model.add(
        tf.keras.layers.Dense(
        num_outputs,
        name="output_dense",
        activation="sigmoid",
        kernel_regularizer=tf.keras.regularizers.l2(hp_l2_reg)
    ))


    # Set optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp_lr)


    if num_outputs > 2:
        print("Using categorical crossentropy")
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    elif num_outputs == 2:
        print("Using binary crossentropy")
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    # compile model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[
            LogitAUC(
                name="logit_auc",
                multi_label=(num_outputs > 2),
            ),
            tf.keras.metrics.AUC(
                curve="ROC",
                name="auc"
            ),
            majority_classifier_acc,
            'accuracy',
            f1_m
        ],
    )
    return model


################################################################################
## Utils/Metrics
################################################################################

def majority_classifier_acc(y_true, y_pred):
    """
    Helper metric function for computing the accuracy that a majority class
    predictor would obtain with the provided labels.
    """
    distr = tf.math.reduce_sum(y_true, axis=0)
    return tf.math.reduce_max(distr) / tf.math.reduce_sum(y_true)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class LogitAUC(tf.keras.metrics.AUC):
    """
    Custom AUC metric that operates in logit activations (i.e. does not
    require them to be positive and will pass a softmax through them before
    computing the AUC)
    """
    def __init__(self, *args, **kwargs):
        super(LogitAUC, self).__init__(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Simply call the parent function with the argmax of the given tensor
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        super(LogitAUC, self).update_state(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )