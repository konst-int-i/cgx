"""
Decompositional implementation of the column generation approach
"""
from . import column_generation_rules
import numpy as np
import os
import pandas as pd
from pathlib import Path
from cgx.utils.metrics import fidelity
from cgx.utils import write_object, flatten, load_object
import swifter
from typing import *
import itertools
from remix.extract_rules.utils import ModelCache
from cgx.explain import BooleanRuleCG, FeatureBinarizer
from tensorflow import keras
from sklearn.metrics import accuracy_score
import logging
from joblib import Parallel, delayed


def extract_cg_decompositional(
        nn: keras.Model,
        x_train: pd.DataFrame,
        y_train: Union[pd.Series, np.array],
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        dataset: str,
        ped_rules,
        sub_threshold: int=10,
        layerwise_substitution: bool=False,
        block_size: int =1):
    """

    Args:
        nn:
        x_train:
        y_train:
        block_size:

    Returns:

    """

    output_layer = len(nn.layers) - 1
    input_hidden_acts = list(range(0, output_layer, block_size))
    output_hidden_acts = input_hidden_acts[1:] + [output_layer]

    num_classes = nn.layers[-1].output_shape[-1]
    total_loop_volume = num_classes * (len(input_hidden_acts) - 1)

    y_pred_nn_train = np.argmax(nn.predict(x_train), axis=1)
    nn_activations = ModelCache(keras_model=nn, train_data=x_train,
                                feature_names=x_train.columns, output_class_names=[str(e) for e in np.sort(np.unique(y_train))])
                                # feature_names=x_train.columns, output_class_names=[str(e) for e in sorted(y_train.unique())])
    logging.info(f"Extracting pedagogical rule set")
    hidden_rule_dict = dict()
    # hidden_rule_objects = []
    subbed_rules = []

    # calculate lookup table for input rule predictions (as candidates to be substituted)
    rule_lookup = substitution_lookup(x_train, k=2, s=sub_threshold, dataset=dataset)

    # explain layer-wise rules
    for hidden_layer, details in enumerate(nn.layers):
        if "output" in details.name:
            continue
        if "dense" not in details.name:
            continue
        # if hidden_layer > 1:  # for debugging
        #     continue
        predictors = nn_activations.get_layer_activations(layer_index=hidden_layer)
        logging.info(f"Extracting rules from hidden layer {hidden_layer} named {details.name}...")
        hidden_rules, _ = column_generation_rules(predictors,
                                            y_pred_nn_train,
                                            cnf = False, # rules for y=1
                                            lambda0 = 0.01,
                                            lambda1 = 0.01,
                                            columns_per_iter=50,
                                            max_degree=10,
                                            beam_width=5,
                                            num_thresh = 10,
                                            negations = True,
                                            iter_max = 25,
                                            verbose = False,
                                            silent = True,
                                            solver="MOSEK",
                                          )
        rule_list = hidden_rules.explain()["rules"]

        y_pred_hidden = hidden_rules.predict(predictors)
        print(f"Hidden {hidden_layer} Fidelity: {fidelity(y_pred_hidden, y_pred_nn_train)}")
        print(f"Hidden {hidden_layer} Accuracy: {accuracy_score(y_pred_hidden, y_train)}")

        if layerwise_substitution:
            # for hidden_rule in hidden_rule_objects:
            subbed_rules.append(substitute_layer(hidden_rules, nn_activations, rule_lookup, predictors, hidden_layer))
        else:
            # substitute all rules
            substituted_rule_dict = dict()
            # for k,v in hidden_rule_dict.items():
            substituted_rule_dict[f"layer_{hidden_layer}"] = [substitute_rule(rule, nn_activations, rule_lookup)
                                                              for rule in rule_list]
            subbed_rules.append(flatten(substituted_rule_dict.values()))

    # TODO - to remove, for development only
    # write single hidden rule object
    write_object(object_=hidden_rules, path="logs/temp/hidden_rules_sample.pkl")
    # write ped_rules
    write_object(object_=ped_rules, path="logs/temp/ped_rules.pkl")
    # write substituted rules
    write_object(object_=subbed_rules, path="logs/temp/subbed_rule_list.pkl")
    # write rule substitution lookup
    write_object(object_=rule_lookup, path="logs/temp/substitution_lookup.pkl")

    # track the currently best accuracy
    current_acc = accuracy_score(y_test, ped_rules.predict(x_test))
    print(f"Current best test acc: {np.round(current_acc, 4)}")
    for new_rule in flatten(subbed_rules):
        # append rule if it improves test performance
        print()
        print(f"Temporarily adding rule {new_rule}")
        ped_rules.append_rule(new_rule, match_existing=False)
        inter_pred = ped_rules.predict(x_test)
        print(ped_rules.explain()["rules"])
        inter_acc = accuracy_score(y_test, inter_pred)
        print(f"Updated acc: {inter_acc}")

        # delete rule again if it doesn't add performance
        if inter_acc <= current_acc:
            print(f"Removing rule {new_rule}")
            ped_rules.remove_last(new_rule)
        else:
            # update the current best accuracy otherwise
            current_acc = inter_acc
        print(ped_rules.explain()["rules"])

    return ped_rules


def substitution_lookup(x_train: pd.DataFrame, k: int, s: int, dataset: str):

    cache_path = Path(f"cache/sub_lookup/{dataset}/lookup_k{k}_s{s}.pkl")
    if os.path.exists(cache_path):
        logging.info(f"Loading input rule lookups for k={k} combinations with s={s}"
                     f"substitution bins from {cache_path}...")
        rule_predictions = load_object(cache_path)
        return rule_predictions

    logging.info(f"Calculating input rule lookups for k={k} combinations with s={s}"
                 f" substitution bins...")
    # Binarise input features to calculate `num_tresh` options
    substitution_binarizer = FeatureBinarizer(numThresh=s, negations=True).fit(pd.DataFrame(x_train))
    x_train_bin = substitution_binarizer.transform(x_train)
    x_train_bin = x_train_bin.T.iloc[:, :0]

    prediction_list = Parallel(n_jobs=-2, prefer="threads")(delayed(_substitution_lookup_helper)(x_train, rules)
                                          for rules in itertools.combinations(list(x_train_bin.index), k))
    prediction_list = [v for v in prediction_list if v is not None]
    rule_predictions = pd.concat(prediction_list, axis=1)

    if not os.path.exists(cache_path):
        os.makedirs(cache_path.parent)
        write_object(rule_predictions, cache_path)

    return rule_predictions

def _substitution_lookup_helper(x_train: pd.DataFrame, rules: List):
    feats = [rule[0] for rule in rules]
    ops = [rule[1] for rule in rules]
    vals = [rule[2] for rule in rules]

    # don't compute if there is any duplicate feature
    if len(feats) != len(set(feats)):
        return None

    predictions = _predict_input_rules(x_train, feats, ops, vals)
    col_name = ""
    for idx, _ in enumerate(feats):
        if idx != 0:
            col_name += " AND "
        col_name += f"{feats[idx]} {ops[idx]} {np.round(vals[idx], 4)}"
    # rule_predictions[col_name] = predictions
    predictions.name = col_name
    return predictions

def _predict_input_rules(x_train: pd.DataFrame, feats: List, ops: List, vals: List) -> pd.Series:
    """
    Takes in two rules and the training dataframes and returns the binary truth values as a series
    corresponding to the rules (i.e., Rule1 AND Rule2)
    Args:
        x_train (pd.DataFrame): Training data
        rule1 (pd.MultiIndex): Clause of MultiIndex format ('feature', 'operation', 'value'), e.g., ('feat1', '>', 0.2423)
        rule2 (pd.MultiIndex): Clause of MultiIndex format ('feature', 'operation', 'value'), e.g., ('feat1', '>', 0.2423)
    Returns:
        pd.Series: series with binary-valued evaluation where the two conditions are true
    """
    sub_df = x_train[feats]
    eval_str = ""
    for idx, _ in enumerate(feats):
        if idx != 0: # add and condition for all but first rule
            eval_str += " & "
        eval_str += f"sub_df.{feats[idx]} {ops[idx]} {vals[idx]}"

    predictions = pd.eval(eval_str, target=sub_df).astype("int")
    return predictions

def substitute_layer(
        hidden_rules: BooleanRuleCG,
        nn_activations:ModelCache,
        rule_lookup: pd.DataFrame,
        predictors: pd.DataFrame,
        layer: int) -> str:
    """
    Substitute all rules from a hidden layer with input rules as an approximation
    Args:
        rules:
        nn_activations:
        rule_lookup:

    Returns:

    """
    logging.info(f"Substituting layer {layer}...")
    # predictors = nn_activations.get_layer_activations(layer_index=int(layer))
    hidden_layer_pred = pd.Series(hidden_rules.predict(predictors), index=rule_lookup.index)

    # hidden_layer_pred.index = sub_lookup.index
    layer_errors = 1 - rule_lookup.sub(hidden_layer_pred, axis="rows").abs()
    layer_errors = 1 - layer_errors.sum(axis=0)/rule_lookup.shape[0]
    layer_errors.sort_values(ascending=True, inplace=True)
    return layer_errors.index[0]


def substitute_rule(rule: str, nn_activations: ModelCache, substitution_lookup: pd.DataFrame) -> str:
    """
    Substitutes a rule from a hidden activation to be approximated in terms of the input features.

    Args:
        rule (str): rule string of format `h_i_j < 0.5`

    Returns:
        str: rule string of format `feat1 < 0.5`
    """
    logging.info(f"Substituting rule {rule}...")
    clauses = rule.split("AND")
    clauses = [c.strip() for c in clauses]

    ret_rule = ""
    for idx, clause in enumerate(clauses):
        feat, operation, value = clause.split(" ")
        _, layer, node = feat.split("_")
        value = float(value)
        operation = ">" if operation == ">" else "<="

        # get activations from layer corresponding to the rule to be substituted
        activations = nn_activations.get_layer_activations(layer_index=int(layer))[feat]
        # get predictions given that activation
        hidden_clause_pred = pd.eval(f"activations {operation} value", target=activations).astype("int")

        # clause_errors = substitution_lookup.apply(lambda x: 1-accuracy_score(x, hidden_clause_pred), axis=0)
        hidden_clause_pred.index = substitution_lookup.index
        clause_errors = 1 - substitution_lookup.sub(hidden_clause_pred, axis="rows").abs()
        clause_errors = 1 - clause_errors.sum(axis=0)/substitution_lookup.shape[0]

        # get rule with lowest error
        clause_errors.sort_values(ascending=True, inplace=True)
        if idx > 0:
            ret_rule += " AND "
        ret_rule += clause_errors.index[0]

    # reconstruct as rule
    logging.info(f"Substituted hidden rule {rule} with {ret_rule}")
    logging.info(f"Substitution error: {np.round(clause_errors[0], 4)}")
    return ret_rule


def _single_rule_predict(x_train, feat, op, val):
    """
    Helper function that returns the binary truth values for each sample in a single candidate rule
    """
    feat = x_train[feat]
    return pd.eval(f"feat {op} {val}").astype("int")
