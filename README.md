# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

This dataset is called **Bank Marketing Data Set** and contains data about marketing campaigns (phone calls) of a Portuguese banking institution. 

It contains 20 input variables related to bank client, last contact information, social and economic attributes and other attributes. The goal is to predict if the client will subscribe a term deposit with the bank. More details can be find at [this link](https://archive.ics.uci.edu/ml/datasets/Bank%20Marketing#).

The best performance model was a VotingEnsemble obtained with the execution of AutoML which resulted in 0.9176 of accuracy.

## Scikit-learn Pipeline

There are two computing resources involved in the Scikit-learn pipeline architecture: an instance used to run the notebook and a cluster used to run Hyperdrive. The notebook create the cluster using the sdk wich is also used to create the container wich will execute Hyperdrive. Once the container is properly started with all dependencies needed to run python with scikit-learn at most 4 simultaneaus jobs from Hyperdrive are executed to do hyperparameter tunning.

The data is read using an Azure Dataset Tabular, created with the from_delimited_files method of the TabularDatasetFactory class from the csv file [from this link](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv).

The classification algorithm used was Logistic Regression and the hyper parameters wich were tunned by Hyperdrive where:
 - C: Inverse of regularization strength (smaller values specify stronger regularization);
 - solver: algorithm used in the optimization;
 - max_iter: Maximum number of iterations taken for the solvers to converge.

The benefits for RandomParameterSampling is the lower resource consume and in most of the cases equivalent performance results comparing to other exhaustive methods.

One of the benefit for the early stopping policy Bandit is that it is more suitable for resource savings.

## AutoML
The VotingEnsemble is a model that combines different classifiers and uses the majority vote or the average of the predicted probabilities to predict the class labels.
Hyperparameters are the ensemble algorithms used, the weights and the number of interactions considered in each classifier.

## Pipeline comparison
The best accuracy obtained with AutoML Voting Ensemble was 0.9176 (using 4 XGBoostClassifiers and 1 LighGBM with differents weights) and with Hyperdrive LogisticRegression classifier was 0.9117 (C=0.1, max_iter=500 and solver='newton-cg').  The difference is small (less than 1%) and is explained by the algorithms differences. 

AutoML results for the same method (LogisticRegression) presented lower accuracy (0.9108) than Hyperdrive approach.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

It might be a good option to check improvements with Hyperdrive approach when using other hyperparameters, stopping policies and sampler.

It is also a good otpion to test improvements using algorithms not covered by AutoML. Neural networks algorithms for example should have results evaluated (HyperDrive is an option for using this type of model).

