from cgx.explain import BooleanRuleCG, FeatureBinarizer
import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
from typing import *
from cgx.rules.rule import Rule
# from cgx.rules.ruleset import Ruleset
from cgx.rules.term import Term
# from cgx.rules.merge import merge
import logging
import warnings
import pickle
warnings.filterwarnings("ignore")


def column_generation_rules(x: pd.DataFrame,
                            y: pd.DataFrame,
                            cnf: bool = True,  # True generates rules for y=1, False for y=0
                            lambda0: float = 0.001,
                            lambda1: float = 0.0001,
                            num_thresh: int = 9,
                            negations = True,
                            iter_max: int = 15,
                            time_max: int = 100,  # max iteration time in seconds
                            columns_per_iter=10,  # max number of columns generated per iteration
                            max_degree=10,  # maximum degree
                            beam_width=5, # beam search width
                            verbose: bool = False,
                            silent: bool = False,
                            solver: str = "MOSEK",  # use since its fully parallelised on CPU
                            **kwargs
                            ):
    """
    Pedagogical method to use boolean rule sets from Column Generation as surrogate model
    Args:
        x:
        y:
        cnf:
        lambda0:
        lambda1:
        num_thresh:
        negations:
        iter_max:
        verbose:
        silent:
        solver:
        **kwargs:

    Returns:
        Tuple:
    """
    # Generate binary columns using binarizer
    binarizer = FeatureBinarizer(numThresh=num_thresh, negations=negations).fit(pd.DataFrame(x))
    x_bin = binarizer.transform(x)

    # Generate rules using linear programming given binarized generated columns
    cg_rules = BooleanRuleCG(CNF=cnf,
                             lambda0=lambda0,
                             lambda1=lambda1,
                             K=columns_per_iter,
                             D=max_degree,
                             B=beam_width,
                             iterMax=iter_max,
                             timeMax=time_max,
                             verbose=verbose,
                             silent=silent,
                             solver=solver,
                             )
    cg_rules.fit(x_bin, y)

    cg_rules.binarizer = binarizer
    cg_rules.regression = False

    # train accuracy
    y_cg_rule_predict = cg_rules.predict(x)
    logging.info(f"Train Accuracy: {np.round(sklearn.metrics.accuracy_score(y, y_cg_rule_predict), 4)}")

    rules = cg_rules.explain()
    # rules corresponding to y=0 (need to create to have a balanced "rule vote")
    # opp_rules = create_opposite_rules(rules)
    logging.info(f"Rules: {rules['rules']}")
    # convert to rule set
    # rules = convert_to_ruleset(rules)
    # opp_rules = convert_to_ruleset(opp_rules)

    # cg_rule_objects = rules.union(opp_rules)
    # cg_rule_objects = merge(cg_rule_objects)
    # cg_rules_converted = Ruleset(rules=cg_rule_objects, feature_names=x.columns)

    # cg_rule_objects = set1.union(set2)
    return rules
    # return cg_rules, cg_rules_converted


def create_opposite_rules(cg_rules: dict) -> dict:
    operator_map = {">": "<=",
                    ">=": "<",
                    "<": ">=",
                    "<=": ">"}

    anti_rules = []
    for rule in cg_rules["rules"]:
        new_terms = []
        for term in rule.split("AND"):
            exp = term.split(" ")
            # remove empty strings
            exp = [e for e in exp if e != ""]
            if len(exp) == 0:
                continue
            operator = exp[1]
            # map the oppositve operator to term
            exp[1] = operator_map[operator]
            # rejoin term into single sting
            new_terms.append(" ".join(exp))

        # join terms as AND rule
        anti_rules.append(" AND ".join(new_terms))
    opp_rules = {
        "isCNF": not (cg_rules['isCNF']),
        "rules": anti_rules
    }
    return opp_rules


def convert_to_ruleset(rules: dict, outputs=None) -> Set:
    """

    Args:
        rules:
        outputs:

    Returns:
        Set[Rule]: set of Rule objects. Note that these still need to be converted to a Ruleset object
    """
    ruleset = set()

    if rules["isCNF"]:
        conclusion = 0
    else:
        conclusion = 1
    for rule in rules["rules"]:
        terms = rule.split("AND")
        # other loop here
        rule_terms = set()
        for term in terms:
            term_split = term.split(" ")
            term_split = [e for e in term_split if e != ""]
            # print(term_split)
            if len(term_split) == 0:
                # print(f"Skipping")
                continue
            variable = term_split[0]
            operator = term_split[1]
            # if operator == "==":  # ruleset object doesn't accept exact values
            #     continue
            threshold = term_split[2]
            rule_terms.add(Term(
                variable=variable,
                operator=operator,
                threshold=float(threshold)
            ))
        ruleset.add(Rule.from_term_set(
            premise=rule_terms,
            conclusion=conclusion,
            confidence=1
        ))
    return ruleset