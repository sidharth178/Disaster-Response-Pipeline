# Disaster-Response-Pipelines

This is the third project of data science nanodegree program offered by udacity.

### Libraries used

This repository is written in Python and HTML and some required Python packages given below .

 1. Pandas
 2. numpy
 3. sklearn
 4. sqlalchemy
 5. flask
 6. nltk
 
 
 ### Files 
 
 1. process_data.py : It is a data ETL pipeline

 2. train_classifier.py : This is a machine learning pipeline

 3. run.py : This is a Flask web app
  
 4. data: This folder contains categories datasets and sample messages .

 5. app: This folder cointains the run.py which iniate the web app .
 
### Overview

 The main goal of this project is to make a web app which is an emergency operator that could be exploit in some emergency time like earthquake and tsunami . This web app classify the disaster text message and transmit it to the responsible entity .
In this model,i have built a machine learning pipeline which categorize emergency text messages from the sender.

### Instructions
 To set up the model and database run these following commands 

   1. python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db  Used this command to       run ETL pipeline
   2. python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl  Used this command to run ML pipeline

 
