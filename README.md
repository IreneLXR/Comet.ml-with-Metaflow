# Homework 4: Comet.ml-with-Metaflow

In this homework, we are asked to do experiments tracking with Comet and tuning hyperparameters, and then generalize the flow. The submitted folder will include two task files, one requirements file, and a README.md with instructions and analysis of results.

## 1. Setup and how to run each task

Please first run:

```commandline
pip install -r requirements
```

Then for task_1.py, you can run:

```commandline
python task_1.py run
```

For task_2.py, you can first run python task_2.py run -- help to get help, then based on the instruction you can run python task_2.py run --solver_for_model 'newton-cg,saga,sag,lbfgs', please use comma to separate each solver and please do not put a space after comma, thank you:

```commandline
python task_2.py run --help
python task_2.py run --solver_for_model 'newton-cg,saga,sag,lbfgs'
```

## 2. Comet.ml panel analysis

For task_1, I'm tuning the solver for logistic regression, using the following four solvers: newton-cg, saga, sag, lbfgs. Then I evaluated the model on validation datasets, and then selected the best one to run on test dataset.

From the screenshot from comet dashboard, we can tell that when we set solver = newton-cg, it will give the best result:



