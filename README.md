# Homework 4: Comet.ml-with-Metaflow

In this homework, we are asked to do experiments tracking with Comet and tuning hyperparameters, and then generalize the flow. The submitted folder will include two task files, one requirements file, and a README.md with instructions and analysis of results.

## 1. Setup and how to run each task

Please first run:

```commandline
pip install -r requirements.txt
```

Then for task_1.py, you can run:

```commandline
python task_1.py run
```

For task_2.py, you can first run python task_2.py run -- help to get help, then based on the instruction you can run python task_2.py run --solver_for_model 'newton-cg,saga,sag,lbfgs', please use comma to separate each solver and please do not put a space after comma, thank you.

```commandline
python task_2.py run --help
python task_2.py run --solver_for_model 'newton-cg,saga,sag,lbfgs'
```

## 2. Comet.ml panel analysis

For task_1, I'm tuning the solver for logistic regression, using the following four solvers: newton-cg, saga, sag, lbfgs. Then I evaluated the model on validation datasets, and then selected the best one to run on test dataset.

From the screenshot from comet dashboard, we can tell that based on the AUC score, we should choose to set solver = newton-cg:

![image](https://user-images.githubusercontent.com/46698580/200077012-d605535f-0b3b-4e2c-8723-c1932112c6ea.png)

Figure.1: main panel

![image](https://user-images.githubusercontent.com/46698580/200077210-94aeb161-f5f5-4593-9f42-4da700f15c8d.png)

Figure.2: ROC curves for four solvers. This graph can be found on the Graphics section of the experiment with tag = join

The following figure shows the performance with model's solver set to be newton-cg to run on test dataset

![image](https://user-images.githubusercontent.com/46698580/200077491-6265c961-81aa-4c5d-9047-29f73ac9719e.png)

Figure.3: Test result. This graph can be found on the Metrics section of the experiment with tag = test

Link to task 1: https://www.comet.com/irenelxr/fre7773-homework-4-task-1/view/new/panels


For task_2, I'm also tuning the solver for the logistic regression. By testing, I run the following command:

```commandline
python task_2.py run --help
python task_2.py run --solver_for_model 'liblinear,sag,saga,lbfgs'
```
Then from the Comet dashboard, we can observe that when we set solver = , it will give the best result:



## 3. Introductions for Comet panel

After you click into the page, the first thing you will see is the panel as shown below:

![image](https://user-images.githubusercontent.com/46698580/200077012-d605535f-0b3b-4e2c-8723-c1932112c6ea.png)

Then you can visit the Experiments tab, in this page, all experiments are listed:

![image](https://user-images.githubusercontent.com/46698580/200078503-1c305974-ef7b-4b13-ab36-2b52dd273fc1.png)

Figure.: Experiments page.

You can visit the experiment with tag = test to see the result of the model run on test dataset. ou can visit the chart, metrics, graphics, and confusion matrix to explore the performance:

![image](https://user-images.githubusercontent.com/46698580/200078615-55974f2c-c5d5-48ad-855f-3e1dea4ade72.png)

Figure.: Dashboard for experiment with tag = test

You can also visit the experiment with tag = join, this page shows the comparsion of the performance of 4 different values on the validation dataset. You can visit the chart, metrics, graphics, and confusion matrix to explore the performance:

![image](https://user-images.githubusercontent.com/46698580/200078717-9086272e-d316-4ad1-af58-f828648137ae.png)

Figure.: Dashboard for experiment with tag = join

Then you can also visit the experiments with tag = validate, those pages show the the performance of 4 different values on the validation dataset corresponding to each solver separately. You can visit the chart, metrics, graphics, and confusion matrix to explore the performance:

![image](https://user-images.githubusercontent.com/46698580/200079320-dec47ecb-ace3-459c-9572-66c00caa349f.png)

Figure.: Dashboard for experiment with tag = validation



