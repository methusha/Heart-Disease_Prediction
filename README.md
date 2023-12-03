# Heart Disease Prediction


## Description
This project is aimed towards creating a predictive machine learning application for rapid heart disease classification, aiming to enhance diagnostic accuracy in healthcare. The Random Forest algorithm, achieving a 91.8% accuracy, was selected as the optimal choice among a variety of predictive models in the scikit-learn library. To allow for user interaction, the application was deployed using Streamlit, an open-source framework tailored for machine learning projects. Despite the moderately high accuracy, ongoing refinement of algorithm parameters is essential before responsibly deploying the tool into healthcare settings. Further research and optimization of these parameters are strongly recommended.


## How to Run the Project
This project can be accessed through the Streamlit Community Cloud. Access this link to run the project:

https://cs6440-heart-disease-prediction.streamlit.app/


## How to Use the Project
Once you run the project, you will see a page resembling a form. Here, you can enter the health metrics of a patient and you will see whether or not they are likely to have heart disease. If you would like to test real-patient data, you can refer to the heart.csv file within this repository. This csv file has patient health metrics and whether they have heart disease or not.


## Architecture Diagram
```mermaid
flowchart LR
    Start --> Stop
```




## Credits
The data used to train the Random Forest Model in this project was composed by the following creators:
* Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
* University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
* University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
* V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

The data can be found here:

https://archive.ics.uci.edu/dataset/45/heart+disease


## Selection of the Predictive Model
Various models from the scikit-learn library were trained to determine which yielded the optimal results for this project. The types of algorithms tested were standard and hypertuned versions of Logistic Regression, Decision Tree, Random Forest, SVC and K-Nearest Neighbours. The models trained in this project can be viewed in this file in this directory:

project_development/heart_disease_model_testing.py
