# Databricks notebook source
#importing a dummy dataset:
path = '/FileStore/credit_risk_data.csv'
data = spark.read.csv(path,header=True,inferSchema=True)
data.display()

# COMMAND ----------

#saving the data to a delta table:
data.write.format('delta').save('/dbfs/user/hive/credit_risk_cleaned')

# COMMAND ----------

#reading the delta table:
delta_df = spark.read.format('delta').load('/dbfs/user/hive/credit_risk_cleaned')
display(delta_df)

# COMMAND ----------

#droping a column and overwriting the table:
delta_df = delta_df.drop('member_id')
delta_df.write.format('delta').mode('overwrite').save('/dbfs/user/hive/credit_risk_cleaned')

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY '/dbfs/user/hive/credit_risk_cleaned'

# COMMAND ----------


