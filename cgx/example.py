import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from cgx.models.fcnn import train_dnn, model_fn
from cgx.explain import column_generation_rules
from cgx.explain import extract_cg_decompositional

def main():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, metrics = train_dnn(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            dataset="breast_cancer",
            fold=1,
            # config=None,
            use_pickle=False,
            # save_model=True # don't overwrite
        )
    print(model.summary())

    train_predictions = np.argmax(model.predict(x_train), axis=1)

    rules = column_generation_rules(
        x_train,
        train_predictions,
        cnf = False, # rules for y=1
        lambda0 = 0.001,
        lambda1 = 0.001,
        columns_per_iter=10,
        max_degree=10,
        beam_width=5,
        num_thresh = 5,
        negations = True,
        iter_max = 25,
        verbose = False,
        silent = False,
        solver="MOSEK",
    )

    print(rules)

    # dec_rules = extract_cg_decompositional(
    #                 nn=model,
    #                 x_train=x_train,
    #                 y_train=y_train,
    #                 x_test=x_test,
    #                 y_test=y_test,
    #                 dataset="breast_cancer",
    #                 ped_rules=rules
    #             )
    #
    # print(dec_rules)






if __name__ == '__main__':
    main()