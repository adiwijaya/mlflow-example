# Import various libraries including matplotlib, sklearn, mlflow
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

import h2o
from h2o.automl import H2OAutoML
h2o.init()


# Import mlflow
import mlflow
import mlflow.sklearn
import mlflow.h2o as h2o

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    # Load Diabetes datasets
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Create pandas DataFrame for sklearn ElasticNet linear_model
    Y = np.array([y]).transpose()
    d = np.concatenate((X, Y), axis=1)
    cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
    data = pd.DataFrame(d, columns=cols)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    htrain = h2o.H2OFrame(train)
    htest = h2o.H2OFrame(test)

    # set the predictor names and the response column name
    # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline

    response = "progression"
    train_x = htrain.columns
    train_x.remove(response)

    test_x = htest.columns
    test_x.remove(response)

    test_y = test[[response]]

    max_runtime_secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10

    # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
    with mlflow.start_run():
        aml = H2OAutoML(max_runtime_secs=max_runtime_secs, balance_classes=False)
        aml.train(x=train_x, y=response, training_frame=htrain, validation_frame=htest)

        result_prediction = aml.predict(htest)
        predicted_qualities = h2o.as_list(result_prediction, use_pandas=True)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out ElasticNet model metrics
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        best_model = aml.leader

        # Set tracking_URI first and then reset it back to not specifying port
        # Note, we had specified this in an earlier cell
        # mlflow.set_tracking_uri(mlflow_tracking_URI)

        # Log mlflow attributes for mlflow UI
        mlflow.log_param("max_runtime_secs", max_runtime_secs)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        # Log artifacts (output files)
        mlflow.h2o.save_model(best_model, "/home/mapr/etc/auto_model")
        mlflow.h2o.log_model(best_model, "automl_model")
