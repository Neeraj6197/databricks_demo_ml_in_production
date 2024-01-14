# Databricks notebook source
#importing the cleaned data:
df = spark.table('credit_risk_cleaned')
df.display()

# COMMAND ----------

#splitting the data to train and test:
from sklearn.model_selection import train_test_split

pd_df = df.toPandas()
X = pd_df.drop('loan_status',axis=1)
y = pd_df['loan_status']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.head()

# COMMAND ----------

# DBTITLE 1,Creating & Logging the Model:
#importing the libraries:
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# COMMAND ----------

#transforming the dataset:
le = LabelEncoder()
cat_cols = X_train.select_dtypes(exclude='number')
for i in cat_cols:
    X_train[i] = le.fit_transform(X_train[i].values.reshape(-1,1))
    X_test[i] = le.transform(X_test[i].values.reshape(-1,1))

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

X_train_sc

# COMMAND ----------

#creating and logging the model:
with mlflow.start_run(run_name='Basic_LR_Run') as run:


    #creating the model:
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train_sc,y_train)
    preds = lr.predict(X_test_sc)

    #Logging the model:
    mlflow.sklearn.log_model(lr,"Log_Reg_Model")

    #Logging the metrics;
    f1 = f1_score(y_test,preds)
    mlflow.log_metric('f1_score',f1)

    run_id = run.info.run_id
    exp_id = run.info.experiment_id

    print("run_id",run_id)
    print("exp_id",exp_id)


# COMMAND ----------

#creating a function to log the parameters of the model:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def log_rf(exp_id,run_name,params,X_train,X_test,y_train,y_test):

    with mlflow.start_run(experiment_id=exp_id,run_name=run_name) as run:
        rf = RandomForestClassifier(**params)
        rf.fit(X_train,y_train)
        preds = rf.predict(X_test)

        #log models:
        mlflow.sklearn.log_model(rf,"random_forest_model")

        #log params:
        mlflow.log_params(params)

        #log metrics:
        mlflow.log_metrics({
            'f1_score':f1_score(y_test,preds),
            'auc_roc_score':roc_auc_score(y_test,preds)
        })

        #Log feature importance:
        importance = (pd.DataFrame(list(zip(pd_df.columns, rf.feature_importances_)), columns=["Feature", "Importance"])
                      .sort_values("Importance", ascending=False))
        importance_path = "/importance.csv"
        importance.to_csv(importance_path,index=False)
        mlflow.log_artifact(importance_path, "feature-importance.csv")

        #Log plot:
        fig, ax = plt.subplots()
        importance.plot.bar(ax=ax)
        plt.title('Feature Importances')
        mlflow.log_figure(fig,'feature_importances.png')

        return run.info.run_id

# COMMAND ----------

#applying the created function and training a new model:
params = {
    "n_estimators": 150,
    "max_depth": 5,
}

log_rf(exp_id, "RF_Second_Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

#applying the created function and training another new model:
params_500 = {
    "n_estimators": 500,
    "max_depth": 10,
}

log_rf(exp_id, "RF_Third_Run", params_500, X_train, X_test, y_train, y_test)

# COMMAND ----------

# DBTITLE 1,Querying Past runs:
#importing mlflow client:
from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

#list all the runs for your experiment
display(client.list_run_infos(exp_id))

# COMMAND ----------

#Pull out a few fields and create a spark DataFrame with it.
runs = spark.read.format("mlflow-experiment").load(exp_id)
runs.display()

# COMMAND ----------

#Pull the last run and take a look at the associated artifacts.
runs_rf = runs.orderBy('start_time',ascending=False).first()
client.list_artifacts(runs_rf.run_id)

# COMMAND ----------

#Return the evaluation metrics for the last run.
client.get_run(runs_rf.run_id).data.metrics

# COMMAND ----------

#Reload the model and take a look at the feature importance.
model = mlflow.sklearn.load_model(f"runs:/{runs_rf.run_id}/random_forest_model")
model.feature_importances_

# COMMAND ----------


