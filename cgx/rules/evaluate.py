import numpy as np
import sklearn
# from cg_explain.rules.ruleset import Ruleset
from remix.rules.ruleset import Ruleset
from cgx.explain import BooleanRuleCG

# from cgx.rules import metrics


def evaluate_ruleset(
    ruleset,
    X_test,
    y_test,
    high_fidelity_predictions=None,
    num_workers=1,
    multi_class=False,
):
    """
    Evaluates the performance of the given set of rules given the provided
    test dataset. It compares this results to a higher fidelity prediction
    of these labels (e.g. coming from a neural network)

    Will generate a dictionary with several statistics describing the nature
    and performance of the given ruleset.

    :param Ruleset ruleset: The set of rules we want to evaluate.
    :param np.ndarray X_test: testing data set for evaluation.
    :param np.ndarray y_test: testing labels of X_test for evaluation.
    :param np.ndarray high_fidelity_predictions: labels predicted for X_test
        using a high fidelity method that is not our ruleset.
    :param int num_workers: maximum number of subprocesses we can span when
        evaluating the input rule set.
    :param bool multi_class: whether or not we are dealing with a multi-class
        problem or not.

    :returns Dict[str, object]: A dictionary containing several statistics and
        metrics of the current run.
    """

    # Make our predictions using our ruleset
    predicted_labels = ruleset.predict(
        X_test,
        num_workers=num_workers,
    )
    if not ruleset.regression:
        # Compute Accuracy
        acc = sklearn.metrics.accuracy_score(y_test, predicted_labels)

        # Compute the AUC using this model. For multiple labels, we average
        # across all labels
        if multi_class:
            auc = 0
        else:
            auc = sklearn.metrics.roc_auc_score(
                y_test,
                predicted_labels,
                multi_class="ovr",
                average='samples',
            )
    else:
        loss = sklearn.metrics.mean_squared_error(y_test, predicted_labels)

    # Compute Fidelity
    if high_fidelity_predictions is not None:
        if ruleset.regression:
            fid = sklearn.metrics.mean_squared_error(
                high_fidelity_predictions,
                predicted_labels,
            )
        else:
            fid = metrics.fidelity(predicted_labels, high_fidelity_predictions)
    else:
        fid = None

    # Compute Comprehensibility
    if isinstance(ruleset, Ruleset):
        comprehensibility_results = metrics.comprehensibility(ruleset)
    elif isinstance(ruleset, BooleanRuleCG):
        comprehensibility_results = metrics.comprehensibility(ruleset.ruleset)

    # And wrap them all together
    if ruleset.regression:
        results = dict(
            mse_fid=fid,
            loss=loss,
        )
    else:
        results = dict(
            acc=acc,
            auc=auc,
            fid=fid,
        )
    results.update(comprehensibility_results)
    return results
