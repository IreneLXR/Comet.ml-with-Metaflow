"""

Simple stand-alone script showing end-to-end training of a regression model using Metaflow. 
This script ports the composable script into an explicit dependency graphg (using Metaflow syntax)
and highlights the advantages of doing so. This script has been created for pedagogical purposes, 
and it does NOT necessarely reflect all best practices.

Please refer to the slides and our discussion for further context.

MAKE SURE TO RUN THIS WITH METAFLOW LOCAL FIRST

"""

import os
# MAKE SURE THESE VARIABLES HAVE BEEN SET
os.environ['COMET_API_KEY']="APhJUciWYinYwUTCb8VlyRWGG"
os.environ['MY_PROJECT_NAME'] = "FRE7773ML2022_HW4_task_1"
assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']
assert 'MY_PROJECT_NAME' in os.environ and os.environ['MY_PROJECT_NAME']
print("Running experiment for project: {}".format(os.environ['MY_PROJECT_NAME']))

from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os

# make sure we are running locally for this
#assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
#assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'

from comet_ml import Experiment
from comet_ml import init
import comet_ml
# Create an experiment with your api key
experiment = Experiment(
            api_key="APhJUciWYinYwUTCb8VlyRWGG",
            project_name="FRE7773ML2022_HW4_task_1",
            workspace="irenelxr"
         )
class LogisticRegressionFlow(FlowSpec):
    """
    SampleRegressionFlow is a minimal DAG showcasing reading data from a file 
    and training a model successfully.
    """
    # load dataset from sklearn.datasets
    # split train, validation, and test dataset
    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.2
    )
    # for validation_size: 0.25 x 0.8 = 0.2
    VALIDATION_SPLIT = Parameter(
        name='validation_split',
        help='Determining the split of the dataset for validation',
        default=0.25
    )

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the sklearn.datasets
        """
        from sklearn.datasets import load_breast_cancer
        self.Xs, self.Ys = load_breast_cancer(return_X_y=True)
        # go to next step
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Xs, 
            self.Ys, 
            test_size=self.TEST_SPLIT, 
            random_state=42
            )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
             self.X_train, 
             self.y_train, 
             test_size=self.VALIDATION_SPLIT, 
             random_state=42) # 0.25 x 0.8 = 0.2
        self.next(self.set_hyperparameters)
        
    @step 
    def set_hyperparameters(self):
        # tunning hyperparameter C
        self.params = ['newton-cg', 'sag', 'saga', 'lbfgs'] 
        self.next(self.train_model, foreach='params')
        
    @step
    def train_model(self):
        """
        Train a regression on the training set
        """
        from sklearn import linear_model
        self.solver = self.input
        experiment.add_tag(self.solver)
        experiment.add_tag("train")
        with experiment.train():
            experiment.log_parameter("solver", self.solver)
            reg = linear_model.LogisticRegression(solver=self.solver)
            reg.fit(self.X_train, self.y_train)
            # now, make sure the model is available downstream
            self.model = reg
        # go to the validation phase
        self.next(self.validation_model)
    
    @step 
    def validation_model(self):
        #Validate the model on the hold out sample
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import RocCurveDisplay
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt
        experiment.add_tag(self.solver)
        experiment.add_tag("validation")
        with experiment.validate():
            self.y_predicted_val = self.model.predict(self.X_val)
            self.score = self.model.score(self.X_val, self.y_val)
            self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.y_val, self.y_predicted_val)
            self.roc_auc = metrics.auc(self.fpr, self.tpr)
            self.metrics = {"solver":self.solver, "validation_score": self.score, "roc_auc_score":self.roc_auc}
            self.info = {"solver": self.solver,"validation_score": self.score, "roc_auc_score":self.roc_auc, "fpr":self.fpr, "tpr":self.tpr, "y_val":self.y_val, "y_predicted_val": self.y_predicted_val}
            self.display = metrics.RocCurveDisplay(fpr=self.fpr, tpr=self.tpr, roc_auc=self.roc_auc)
            plt.plot(self.fpr, self.tpr, label='AUC='+str(self.roc_auc))
            plt.legend(loc=4)
            plt.show()
            # log scores, confusion matrix and plot
            experiment.log_confusion_matrix(self.y_val, self.y_predicted_val)
            experiment.log_metrics(self.metrics)
            experiment.set_model_graph(plt)
            experiment.log_figure(figure_name="Figure", figure=plt)
        # all is done go to the end
        self.next(self.join)
     
    @step
    def join(self, inputs):
        # join
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import RocCurveDisplay
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt
        experiment.add_tag("join")
        # Logs which model was the best to the Run Experiment to easily
        # compare between different Runs
        self.infos = [input.info for input in inputs]
        self.best_model = max(self.infos, key=lambda info: info["roc_auc_score"])
        self.best_param = self.best_model["solver"]
        self.best_y_val = self.best_model["y_val"]
        self.best_y_pred_val = self.best_model["y_predicted_val"]
        for input in inputs:
            experiment.log_metric("%s_roc_auc_score" % input.solver, input.roc_auc)
            plt.plot(input.fpr, input.tpr, label='AUC_'+input.solver+"="+str(input.roc_auc))
            plt.legend(loc=4)
            plt.show()
        # log scores, confusion matrix and plot
        experiment.log_confusion_matrix(y_true=self.best_y_val, y_predicted=self.best_y_pred_val,titile="Confusion Matrix for " +self.best_param)
        experiment.set_model_graph(plt)
        experiment.log_figure(figure_name="Figure", figure=plt)
        experiment.log_parameter("Best Model", self.best_param)
        # X_train, X_test, y_train, y_test is all the same for the four branches, so just pick the first one
        self.X_train = inputs[0].X_train
        self.y_train = inputs[0].y_train
        self.X_test = inputs[0].X_test
        self.y_test = inputs[0].y_test
        self.next(self.test_model)
    
    @step 
    def test_model(self):
        #Test the model on the hold out sample
        from sklearn import linear_model
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import RocCurveDisplay
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt
        experiment.add_tag("test")
        with experiment.test():
            self.best_model = linear_model.LogisticRegression(solver=self.best_param).fit(self.X_train, self.y_train)
            # now, make sure the model is available downstream
            self.y_predicted_test = self.best_model.predict(self.X_test)
            self.test_score = self.best_model.score(self.X_test, self.y_test)
            self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.y_test, self.y_predicted_test)
            self.roc_auc = metrics.auc(self.fpr, self.tpr)
            self.display = metrics.RocCurveDisplay(fpr=self.fpr, tpr=self.tpr, roc_auc=self.roc_auc)
            plt.plot(self.fpr, self.tpr, label='AUC='+str(self.roc_auc))
            plt.legend(loc=4)
            plt.show()
            self.metrics = {"solver": self.best_param,"test_score": self.test_score, "roc_auc_score":self.roc_auc}
            # log scores, confusion matrix and plot
            experiment.log_confusion_matrix(self.y_test, self.y_predicted_test)
            experiment.log_metrics(self.metrics)
            experiment.set_model_graph(plt)
            experiment.log_figure(figure_name="Figure", figure=plt)
        # all is done go to the end
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        experiment.add_tag("end")
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    LogisticRegressionFlow()
