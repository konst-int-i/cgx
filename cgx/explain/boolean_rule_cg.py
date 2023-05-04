import numpy as np
import pandas as pd
import cvxpy as cvx
from sklearn.base import BaseEstimator, ClassifierMixin
import time

from .beam_search import beam_search, beam_search_K1


class BooleanRuleCG(BaseEstimator, ClassifierMixin):
    """BooleanRuleCG is a directly interpretable supervised learning method
    for binary classification that learns a Boolean rule in disjunctive
    normal form (DNF) or conjunctive normal form (CNF) using column generation (CG).
    AIX360 implements a heuristic beam search version of BRCG that is less
    computationally intensive than the published integer programming version [#NeurIPS2018]_.

    References:
        .. [#NeurIPS2018] `S. Dash, O. Günlük, D. Wei, "Boolean decision rules via
           column generation." Neural Information Processing Systems (NeurIPS), 2018.
           <https://papers.nips.cc/paper/7716-boolean-decision-rules-via-column-generation.pdf>`_
    """
    def __init__(self,
        lambda0=0.001,
        lambda1=0.001,
        CNF=False,
        iterMax=100,
        timeMax=100,
        K=10,
        D=10,
        B=5,
        eps=1e-6,
        solver='ECOS',
        verbose=False,
        silent=False):
        """
        Args:
            lambda0 (float, optional): Complexity - fixed cost of each clause
            lambda1 (float, optional): Complexity - additional cost for each literal
            CNF (bool, optional): CNF instead of DNF
            iterMax (int, optional): Column generation - maximum number of iterations
            timeMax (int, optional): Column generation - maximum runtime in seconds
            K (int, optional): Column generation - maximum number of columns generated per iteration
            D (int, optional): Column generation - maximum degree
            B (int, optional): Column generation - beam search width
            eps (float, optional): Numerical tolerance on comparisons
            solver (str, optional): Linear programming - solver
            verbose (bool, optional): Linear programming - verboseness
            silent (bool, optional): Silence overall algorithm messages
        """
        # Complexity parameters
        self.lambda0 = lambda0      # fixed cost of each clause
        self.lambda1 = lambda1      # additional cost per literal
        # CNF instead of DNF
        self.CNF = CNF
        # Column generation parameters
        self.iterMax = iterMax      # maximum number of iterations
        self.timeMax = timeMax      # maximum runtime in seconds
        self.K = K                  # maximum number of columns generated per iteration
        self.D = D                  # maximum degree
        self.B = B                  # beam search width
        # Numerical tolerance on comparisons
        self.eps = eps
        # Linear programming parameters
        self.solver = solver        # solver
        self.verbose = verbose      # verboseness
        # Silence output
        self.silent = silent

    def fit(self, X, y):
        """Fit model to training data.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
            y (array): Binary-valued target variable
        Returns:
            BooleanRuleCG: Self
        """
        if not self.silent:
            print('Learning {} rule with complexity parameters lambda0={}, lambda1={}'\
                  .format('CNF' if self.CNF else 'DNF', self.lambda0, self.lambda1))
        if self.CNF:
            # Flip labels for CNF
            y = 1 - y
        # Positive (y = 1) and negative (y = 0) samples
        P = np.where(y > 0.5)[0]
        Z = np.where(y < 0.5)[0]

        if len(Z) == 0:
            Z = np.append(Z, P[-1])
            P = P[:-1]
        nP = len(P)
        n = len(y)

        # Initialize with empty and singleton conjunctions, i.e. X plus all-ones feature
        # Feature indicator and conjunction matrices
        z = pd.DataFrame(np.eye(X.shape[1], X.shape[1]+1, 1, dtype=int), index=X.columns)
        A = np.hstack((np.ones((X.shape[0],1), dtype=int), X))
        # Iteration counter
        self.it = 0
        # Start time
        self.starttime = time.time()

        # Formulate master LP
        # Variables
        w = cvx.Variable(A.shape[1], nonneg=True)
        xi = cvx.Variable(nP, nonneg=True)
        # Objective function (no penalty on empty conjunction)
        cs = self.lambda0 + self.lambda1 * z.sum().values # change objective function to introduce further penalties
        cs[0] = 0
        # print(P, Z, nP, n, z, A, w, xi, cs)
        obj = cvx.Minimize(cvx.sum(xi) / n + cvx.sum(A[Z,:] * w) / n + cs * w)
        # Constraints
        constraints = [xi + A[P,:] * w >= 1]

        # Solve problem
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=self.solver, verbose=self.verbose)
        if not self.silent:
            print('Initial LP solved')


        # Extract dual variables
        r = np.ones_like(y, dtype=float) / n
        # if constraints[0].dual_value is None:
        #     pass
        # else:
        r[P] = -constraints[0].dual_value

        # Beam search for conjunctions with negative reduced cost
        # Most negative reduced cost among current variables
        UB = np.dot(r, A) + cs
        UB = min(UB.min(), 0)
        v, zNew, Anew = beam_search(r, X, self.lambda0, self.lambda1,
                                    K=self.K, UB=UB, D=self.D, B=self.B, eps=self.eps)

        while (v < -self.eps).any() and (self.it < self.iterMax) and (time.time()-self.starttime < self.timeMax):
            # Negative reduced costs found
            self.it += 1
            if not self.silent:
                print('Iteration: {}, Objective: {:.4f}'.format(self.it, prob.value))

            # Add to existing conjunctions
            z = pd.concat([z, zNew], axis=1, ignore_index=True)
            A = np.concatenate((A, Anew), axis=1)

            # Reformulate master LP
            # Variables
            w = cvx.Variable(A.shape[1], nonneg=True)
            # Objective function
            cs = np.concatenate((cs, self.lambda0 + self.lambda1 * zNew.sum().values))
            obj = cvx.Minimize(cvx.sum(xi) / n + cvx.sum(A[Z,:] * w) / n + cs * w)
            # Constraints
            constraints = [xi + A[P,:] * w >= 1]

            # Solve problem
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=self.solver, verbose=self.verbose)

            # Extract dual variables
            r[P] = -constraints[0].dual_value

            # Beam search for conjunctions with negative reduced cost
            # Most negative reduced cost among current variables
            UB = np.dot(r, A) + cs
            UB = min(UB.min(), 0)
            v, zNew, Anew = beam_search(r, X, self.lambda0, self.lambda1,
                                        K=self.K, UB=UB, D=self.D, B=self.B, eps=self.eps)

        # Save generated conjunctions and LP solution
        self.z = z
        self.wLP = w.value

        r = np.full(nP, 1./n)
        # K=1 beam search to generate single solution
        self.w = beam_search_K1(r, pd.DataFrame(1-A[P,:]), 0, A[Z,:].sum(axis=0) / n + cs,
                                UB=r.sum(), D=100, B=2*self.B, eps=self.eps, stopEarly=False)[1].values.ravel()
        if len(self.w) == 0:
            self.w = np.zeros_like(self.wLP, dtype=int)

    def compute_conjunctions(self, X):
        """Compute conjunctions of features as specified in self.z.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
        Returns:
            array: A -- Conjunction values
        """
        try:
            A = 1 - (np.dot(1 - X, self.z) > 0)
        except AttributeError:
            print("Attribute 'z' does not exist, please fit model first.")
        return A

    def predict(self, X, **kwargs):
        """Predict class labels.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
        Returns:
            array: y -- Predicted labels
        """
        # Check convert features if not binarized
        if hasattr(self, "binarizer"):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.ruleset.feature_names)
            X = self.binarizer.transform(X)

        # Compute conjunctions of features
        A = self.compute_conjunctions(X)
        # Predict labels
        if self.CNF:
            # Flip labels since model is actually a DNF for Y=0
            return 1 - (np.dot(A, self.w) > 0)
        else:
            return (np.dot(A, self.w) > 0).astype(int)

    def explain(self, maxConj=None, prec=2):
        """Return rules comprising the model.

        Args:
            maxConj (int, optional): Maximum number of conjunctions to show
            prec (int, optional): Number of decimal places to show for floating-value thresholds
        Returns:
            Dictionary containing

            * isCNF (bool): flag signaling whether model is CNF or DNF
            * rules (list): selected conjunctions formatted as strings
        """
        # Selected conjunctions
        z = self.z.loc[:, self.w > 0.5]
        truncate = (maxConj is not None) and (z.shape[1] > maxConj)
        nConj = maxConj if truncate else z.shape[1]

        """
        if self.CNF:
            print('Predict Y=0 if ANY of the following rules are satisfied, otherwise Y=1:')
        else:
            print('Predict Y=1 if ANY of the following rules are satisfied, otherwise Y=0:')
        """

        # Sort conjunctions by increasing order
        idxSort = z.sum().sort_values().index[:nConj]
        # Iterate over sorted conjunctions
        conj = []
        for i in idxSort:
            # MultiIndex of features participating in rule i
            idxFeat = z.index[z[i] > 0]
            # String representations of features
            strFeat = idxFeat.get_level_values(0) + ' ' + idxFeat.get_level_values(1)\
                + ' ' + idxFeat.get_level_values(2).to_series()\
                .apply(lambda x: ('{:.' + str(prec) + 'f}').format(x) if type(x) is float else str(x))
            # String representation of rule
            strFeat = strFeat.str.cat(sep=' AND ')
            conj.append(strFeat)

        return {
            'isCNF': self.CNF,
            'rules': conj
        }

    def append_rule(self, rule: str, match_existing: bool = True, negations: bool=True) -> None:
        """
        Update the ``w`` and ``z`` using a new rule that can be specified of the format ``feature operator value``,
        e.g., ``feat1 < 0.5``
        Note that not the exact same rule is added to the rule set, but the closest match to the cutoff value
        from the binarized columns.

        Args:
            rule (str): rule in string format
            match_existing (bool): If true, looks for closest match of existing binarised
                rules. If false, adds new index corresponding to the exact rule added.
                Use False for hidden layers, otherwise it's unlikely to improve the
                pedagogical solution.

        Returns:
            None: does not return anything, but updating w and z will update the behaviour of ``.explain()`` and
            ``.predict()``
        """

        invalid_format = "Invalid format provided, please specify rule of format `feat1 < 0.5`."

        new_idx = self.z.columns[-1] + 1  # index used in `self.w` and `self.z` to store new rule
        self.z.loc[:, new_idx] = 0

        # split rule into subclauses
        clauses = rule.split("AND")
        assert len(clauses) >= 1, invalid_format
        clauses = [c.strip() for c in clauses]

        for clause in clauses:
            assert len(clause.split(" ")) == 3, invalid_format
            feature, operation, value = clause.split(" ")
            value = float(value)
            operation = ">" if operation == ">" else "<="
            neg_operation = ">" if operation == "<=" else "<="

            if match_existing:
                # get binarized cutoff values to determine closest match to the value of the rule to be added
                feature_cutoffs = np.array(
                    self.z.loc[(self.z.index.get_level_values("feature") == feature)].index.get_level_values("value").unique()
                )
                # get the closest match for the bin cutoff values (absolute difference)
                match = feature_cutoffs[np.argmin(abs(feature_cutoffs - value))]

                # assign new rule
                self.z.loc[(f'{feature}', f'{operation}', match), new_idx] = 1
            else:
                # create new rule in row-index
                self.z.loc[(f'{feature}', f'{operation}', value), :] = 0
                if negations:  # ensure that shapes of w, z, and binarized(X) match
                    self.z.loc[(f'{feature}', f'{neg_operation}', value), :] = 0
                # activate rule in column-index
                self.z.loc[(f'{feature}', f'{operation}', value), new_idx] = 1
                # update the FeatureBinarizer for update to work on `predict` method
                self.binarizer.update(feature, value)

        # make sure it's logged in the rule mask
        self.w = np.append(self.w, 1)


        return None

    def remove_last(self, rule, negations: bool=True) -> None:
        """
        Removes last rule from the currently active rules
        Returns:
            None: Updates self.w and self.z accordingly
        """
        invalid_format = "Invalid format provided, please specify rule of format `feat1 < 0.5`."
        # split rule into subclauses
        clauses = rule.split("AND")
        clauses = [c.strip() for c in clauses]

        for clause in clauses:
            assert len(clause.split(" ")) == 3, invalid_format
            feature, operation, value = clause.split(" ")
            value = float(value)
            operation = ">" if operation == ">" else "<="
            neg_operation = ">" if operation == "<=" else "<="

            # drop indeces from z to match shapes
            self.z.drop((f'{feature}', f'{operation}', value), inplace=True)
            if negations:
                self.z.drop((f'{feature}', f'{neg_operation}', value), inplace=True)

            self.binarizer.remove(feature, value)

        self.z = self.z.iloc[:, :-1]
        self.w = self.w[:-1]
        return None

    def remove_all(self) -> None:
        """
        Removes all rules from the object
        """
        dims = len(self.w)
        self.w = np.zeros(dims)
        return None


