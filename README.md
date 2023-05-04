# cgx

Implementation of ICLR 2023 Workshop paper: [CGXplain: Rule-Based Deep Neural Network Explanations Using Dual Linear Programs](https://openreview.net/pdf?id=bHbf5-nE8N)

Accepted at ICLR Workshop on Trustworthy ML for Healthcare ([TML4H](https://sites.google.com/view/tml4h2023/home?authuser=0)). 

## Setup
Run the CLI commands to install required dependencies. Note that this setup requires `conda`, but we recommend 
using [`mamba`](https://mamba.readthedocs.io/en/latest/installation.html) for a faster installation.

```shell
conda env create -f environment.yml
conda activate cgx
```

## Basic Usage

Train a basic neural network. We currently support tensorflow, but plan to add PyTorch support soon.
We provide some training utils, but any `tensorflow` model can be used to extract rules from. 


### Pedagogical Example
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from cgx.models.fcnn import train_dnn, model_fn
from cgx.explain import column_generation_rules

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model, metrics = train_dnn(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
print(model.summary())

train_predictions = np.argmax(model.predict(x_train), axis=1)

rules = column_generation_rules(
    x_train,
    train_predictions,
    cnf = False, 
    lambda0 = 0.001, # penalty for # rules
    lambda1 = 0.001, # penalty for # terms
)
```

Output
```shell
'rules': ['worst radius <= 14.97', 
          'concave points error > 0.01 AND worst concavity <= 0.15',
          'mean compactness > 0.06 AND worst texture <= 27.83 AND worst area <= 862.07 AND worst smoothness <= 0.15 AND worst symmetry <= 0.33', 
          'mean radius > 13.30 AND mean fractal dimension > 0.06 AND area error <= 24.72 AND worst smoothness > 0.11 AND worst symmetry <= 0.30']
```         

