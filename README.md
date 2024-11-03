# **Automated Review Classification Pipeline with BERTopic**

This repository contains an end-to-end machine learning pipeline for automating the classification of app reviews from the Google Play Store using the BERTopic model. The project integrates Databricks, AWS S3, and a Streamlit dashboard, and leverages a combination of PySpark, Python, and data engineering practices for scalability and automation.

## **Architecture**

![Project Architecture](https://github.com/DobreMihai04/ReviewsTopic/blob/main/images/example_image_1.png)


## **Project Overview**

The Review Classification Pipeline automatically:
- Scrapes app reviews daily from the Google Play Store.
- Classifies topics from reviews using a fine-tuned BERTopic model.
- Pre-aggregates results and saves them in AWS S3.
- Visualizes the results via a Streamlit dashboard.

The pipeline is designed to run on a daily schedule via Databricks workflows, ensuring fresh data is processed and available for visualization every day. The project uses AWS S3 as the central storage solution for both raw and processed data.


## **Technologies Used**
- **Python** (Pandas, PySpark,Google Play Scraper)
- **Databricks** for scheduling daily workflows and notebooks.
- **BERTopic** for topic modeling and classification.
- **Hugging Face** for hosting the BERTopic model.
- **AWS S3** for data storage.
- **Streamlit** for data visualization.
- **AWS EC2** for dashboard hosting.

---

## **Components**

### **1. Model Training**

- **Training Notebook**:
  - Trains and fine-tunes the BERTopic model for review classification.
  - Uses historical app review data.
  - Pushes the trained model to the Hugging Face Hub.

### **2. Databricks Notebooks and Workflow**
The Databricks Workflow runs daily to automate the following steps:

- **Scraping Notebook**:
  - Scrapes new reviews from the Google Play Store daily.
  - Saves raw data as CSV in AWS S3.

- **Processing Notebook**:
  - Reads raw review data from S3 and processes it using PySpark.
  - Converts the processed data into a Delta Table for efficient querying.

- **Model Inference Notebook**:
  - Loads the BERTopic model from Hugging Face.
  - Classifies the topics of new reviews.
  - Saves the topic predictions back to S3 as Delta Tables.

- **Pre-Aggregation Notebook**:
  - Prepares aggregated results for visualization.
  - Saves pre-aggregated tables as CSVs in S3.

### **3. Streamlit Dashboard**
- **Streamlit Dashboards**:
  - Pulls pre-aggregated data from S3.
  - Visualizes topics, reviews, and trends using interactive charts.

---

## Running it
*the backend requires AWS credentials since the models are stored on S3

### Setup:
1.  Clone this repository

2.  Create a python3 virtual enviroment
     ```  
    python3 -m venv .env
    ```
3.  Activate the virtual enviroment
    ```   
    source .env/bin/activate
    ```
4.  Install dependencies
    ```  
    pip3 install -r requirements.txt
    ```

### Run: 

#### 1. Model Training
You can train the BERTopic model and tune topic representation in a docker container using the provided Jupyter notebook in the `model-training/` folder:

Run docker container:


    cd model-training
    
    docker-compose up
    



#### 2. Streamlit Dashboard
  To run the Streamlit dashboard locally:

    
    cd streamlit
    
    streamlit run app.py