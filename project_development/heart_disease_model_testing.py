# -*- coding: utf-8 -*-
"""

@author: methusha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

heart_df = pd.read_csv('heart.csv')

heart_df.info()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

heart_df['Sex']= label_encoder.fit_transform(heart_df['Sex'])
heart_df['ChestPainType']= label_encoder.fit_transform(heart_df['ChestPainType'])
heart_df['RestingECG']= label_encoder.fit_transform(heart_df['RestingECG'])
heart_df['ExerciseAngina']= label_encoder.fit_transform(heart_df['ExerciseAngina'])
heart_df['ST_Slope']= label_encoder.fit_transform(heart_df['ST_Slope'])

from sklearn.model_selection import train_test_split

x = heart_df.drop('HeartDisease',axis=1)
y = heart_df['HeartDisease']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

samples = x.shape[0]
features = x.shape[1]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#%%

#Logistic Regression

from sklearn.linear_model import LogisticRegression

#Define the model
log_reg = LogisticRegression(solver='liblinear', max_iter=1000)

#Fit the model
log_reg.fit(x_train, y_train)

#Predict for the test set
y_pred = log_reg.predict (x_test)

y_probabilities = log_reg.predict_proba(x_test)

from sklearn import metrics

#Retrieve the confusion matrix
log_confusion_mat = metrics.confusion_matrix(y_test, y_pred)

#Call the confusion matrix
log_confusion_mat

#Plot the Conf. Mat. as a heatmap
import seaborn as sns

axis_ticks=[0,1] # name  of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

#im = ax.imshow(confusion_mat, cmap="Pastel1", aspect="auto")
sns.heatmap(pd.DataFrame(log_confusion_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('-----!! Confusion Matrix for Heart Disease' 
          '- Logistic Regression !!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

log_reg_acc_score = log_reg.score(x_test,y_test)
log_reg_acc_score

#%%

#Logistic Regression, with Hypertuning

from sklearn.model_selection import GridSearchCV

param_grid = {'penalty': ['l1','l2'],
              'C': [0.001,0.01,0.1,1,10,100,1000],
              'fit_intercept': [True,False]}
log_grid = GridSearchCV(log_reg, param_grid)

log_grid.fit(x_train, y_train)

log_grid.best_params_

best_model =log_grid.best_estimator_
best_model

y_fit = best_model.predict(x_test)

#Define the hypertuned model
ht_log_reg = LogisticRegression(C=10, max_iter=1000, penalty='l2', solver='liblinear')

#Fit the model
ht_log_reg.fit(x_train, y_train)

#Predict for the test set
ht_y_pred = ht_log_reg.predict (x_test)

ht_y_probabilities = ht_log_reg.predict_proba(x_test)


#Retrieve the confusion matrix
ht_log_confusion_mat = metrics.confusion_matrix(y_test, ht_y_pred)

#Call the confusion matrix
ht_log_confusion_mat


axis_ticks=[0,1] # name  of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

#im = ax.imshow(confusion_mat, cmap="Pastel1", aspect="auto")
sns.heatmap(pd.DataFrame(ht_log_confusion_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('-----!! Confusion Matrix for Heart Disease' 
          '- Logistic Regression (Hypertuned)!!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

ht_log_confusion_mat = metrics.confusion_matrix(y_test, y_fit)

ht_log_reg_acc_score = ht_log_reg.score(x_test,y_test)
ht_log_reg_acc_score

#%%

#Decision Tree

from sklearn.tree import DecisionTreeClassifier

DTF_clf = DecisionTreeClassifier()

DTF_clf.fit(x_train,y_train)

DT_predictions = DTF_clf.predict(x_test)

DT_conf_mat = metrics.confusion_matrix(y_test,DT_predictions)

DT_conf_mat

axis_ticks=[0,1] # name  of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

#im = ax.imshow(confusion_mat, cmap="Pastel1", aspect="auto")
sns.heatmap(pd.DataFrame(DT_conf_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('-----!! Confusion Matrix for Heart Disease' 
          '- Decision Tree!!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

dec_tree_acc_score = DTF_clf.score(x_test,y_test)
dec_tree_acc_score

#%%

#Decision Tree, with Hypertuning

clfr = DecisionTreeClassifier()

parameters = {'criterion':['gini', 'entropy'], 'splitter':['best','random'],
              'min_samples_split':np.arange(2, 100)}

dt_grid = GridSearchCV(clfr, parameters)
dt_grid.fit(x_train,y_train)

dt_grid.best_params_

best_model = dt_grid.best_estimator_
best_model

DTh_clf = best_model

DTh_clf.fit(x_train,y_train)

DTh_predictions = DTh_clf.predict(x_test)

DTh_conf_mat = metrics.confusion_matrix(y_test,DTh_predictions)

DTh_conf_mat

axis_ticks=[0,1] # name  of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

#im = ax.imshow(confusion_mat, cmap="Pastel1", aspect="auto")
sns.heatmap(pd.DataFrame(DTh_conf_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('-----!! Confusion Matrix for Heart Disease' 
          '- Decision Tree (Hypertuned)!!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

ht_dec_tree_acc_score = DTh_clf.score(x_test,y_test)
ht_dec_tree_acc_score

#%%

#Random Forest

from sklearn.ensemble import RandomForestClassifier

RF_clf = RandomForestClassifier()

RF_clf.fit(x_train,y_train)

RF_predictions = RF_clf.predict(x_test)

RF_conf_mat = metrics.confusion_matrix(y_test,RF_predictions)

RF_conf_mat

axis_ticks=[0,1] # name  of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

#im = ax.imshow(confusion_mat, cmap="Pastel1", aspect="auto")
sns.heatmap(pd.DataFrame(RF_conf_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('-----!! Confusion Matrix for Heart Disease' 
          '- Random Forest!!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

ran_for_acc_score = RF_clf.score(x_test,y_test)
ran_for_acc_score

#%%

#Random Forest, with Hypertuning

RF_clfr = RandomForestClassifier()

RF_parameters = {'criterion':['gini', 'entropy'], 'bootstrap':[True,False],
              'n_estimators':np.arange(10,1010,100)}

rf_grid = GridSearchCV(RF_clfr, RF_parameters)
rf_grid.fit(x_train,y_train)

rf_grid.best_params_

best_model = rf_grid.best_estimator_
best_model

RFh_clf = best_model

RFh_clf.fit(x_train,y_train)

RFh_predictions = RFh_clf.predict(x_test)

RFh_conf_mat = metrics.confusion_matrix(y_test,RFh_predictions)

RFh_conf_mat

axis_ticks=[0,1] # name  of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

#im = ax.imshow(confusion_mat, cmap="Pastel1", aspect="auto")
sns.heatmap(pd.DataFrame(RFh_conf_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('-----!! Confusion Matrix for Heart Disease' 
          '- Random Forest (Hypertuned)!!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

ht_ran_for_acc_score = RFh_clf.score(x_test,y_test)

#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Hyperparameter Tuning
RF_clfr = RandomForestClassifier()
RF_parameters = {
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False],
    'n_estimators': np.arange(10, 1010, 25)
}

rf_grid = GridSearchCV(RF_clfr, RF_parameters, cv=5)
rf_grid.fit(x_train_scaled, y_train)

# Get the best parameters
best_params = rf_grid.best_params_

# Create a new Random Forest model with the best parameters
best_model = RandomForestClassifier(**best_params)
best_model.fit(x_train_scaled, y_train)

# Feature Importance Analysis
feature_importances = best_model.feature_importances_
# Print or analyze feature_importances here

# Evaluate on the Test Set
RFh_predictions = best_model.predict(x_test_scaled)
RFh_conf_mat = metrics.confusion_matrix(y_test, RFh_predictions)

# Plot Confusion Matrix
axis_ticks = [0, 1]  # name of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

sns.heatmap(pd.DataFrame(RFh_conf_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
plt.title('-----!! Confusion Matrix for Heart Disease - Random Forest (Hypertuned) !!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Accuracy Score
ht_ran_for_acc_score = best_model.score(x_test_scaled, y_test)




#%%

#Support Vector Machine

from sklearn.svm import SVC

SVC_clf = SVC()

SVC_clf.fit(x_train, y_train)

SVC_predictions = SVC_clf.predict(x_test)

SVC_conf_mat = metrics.confusion_matrix(y_test,SVC_predictions)

SVC_conf_mat

axis_ticks=[0,1] # name  of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

#im = ax.imshow(confusion_mat, cmap="Pastel1", aspect="auto")
sns.heatmap(pd.DataFrame(SVC_conf_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('-----!! Confusion Matrix for Heart Disease' 
          '- SVC!!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

SVC_acc_score = SVC_clf.score(x_test,y_test)

#%%

#Support Vector Machine, with Hypertuning

SVC_clfr = SVC()

SVC_parameters = {'C':[1,5,10,50],
                  "gamma":[0.0001,0.0005,0.001,0.005],
                  }

svc_grid = GridSearchCV(SVC_clfr, SVC_parameters)
svc_grid.fit(x_train,y_train)

svc_grid.best_params_

best_model = svc_grid.best_estimator_
best_model

SVCh_clf = best_model

SVCh_clf.fit(x_train, y_train)

SVCh_predictions = SVCh_clf.predict(x_test)

SVCh_conf_mat = metrics.confusion_matrix(y_test,SVCh_predictions)

SVCh_conf_mat

axis_ticks=[0,1] # name  of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

#im = ax.imshow(confusion_mat, cmap="Pastel1", aspect="auto")
sns.heatmap(pd.DataFrame(SVCh_conf_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('-----!! Confusion Matrix for Heart Disease' 
          '- SVC (Hypertuned)!!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

ht_SVC_acc_score = SVCh_clf.score(x_test,y_test)
ht_SVC_acc_score

#%%

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.kernel_approximation import Nystroem
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Use a smaller subset of your data for hyperparameter tuning
x_train_subset, _, y_train_subset, _ = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Feature Scaling on the subset
scaler = StandardScaler()
x_train_subset_scaled = scaler.fit_transform(x_train_subset)
x_test_scaled = scaler.transform(x_test)

# Kernel Approximation
nystroem = Nystroem(kernel='rbf', n_components=100, random_state=42)
x_train_subset_approx = nystroem.fit_transform(x_train_subset_scaled)
x_test_scaled_approx = nystroem.transform(x_test_scaled)

# Hyperparameter Tuning with Parallel Processing
SVC_clfr = SVC()
SVC_parameters = {
    'C': [0.1, 1, 10, 50, 100, 500],
    'gamma': [0.0001, 0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5, 1.0]
}

svc_grid = GridSearchCV(SVC_clfr, SVC_parameters, cv=5, n_jobs=-1)
svc_grid.fit(x_train_subset_approx, y_train_subset)

# Get the best parameters
best_params = svc_grid.best_params_

# Create a new SVM model with the best parameters
best_model = SVC(**best_params)
best_model.fit(x_train_subset_approx, y_train_subset)

# Evaluate on the Test Set
SVCh_predictions = best_model.predict(x_test_scaled_approx)
SVCh_conf_mat = metrics.confusion_matrix(y_test, SVCh_predictions)

# Plot Confusion Matrix
axis_ticks = [0, 1]  # name of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

sns.heatmap(pd.DataFrame(SVCh_conf_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
plt.title('-----!! Confusion Matrix for Heart Disease - SVC (Hypertuned) !!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Accuracy Score
ht_SVC_acc_score = best_model.score(x_test_scaled_approx, y_test)




#%%

#K-Nearest Neighbours

from sklearn.neighbors import KNeighborsClassifier

KNN_clf = KNeighborsClassifier()

KNN_clf.fit(x_train, y_train)

KNN_predictions = KNN_clf.predict(x_test)

KNN_conf_mat = metrics.confusion_matrix(y_test,KNN_predictions)

KNN_conf_mat

axis_ticks=[0,1] # name  of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

#im = ax.imshow(confusion_mat, cmap="Pastel1", aspect="auto")
sns.heatmap(pd.DataFrame(KNN_conf_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('-----!! Confusion Matrix for Heart Disease' 
          '- K-Nearest Neigbours!!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

KNN_acc_score = KNN_clf.score(x_test,y_test)
KNN_acc_score

#%%

#K-Nearest Neighbours, with Hypertuning

KNN_clfr = KNeighborsClassifier()

KNN_parameters = {"n_neighbors":np.arange(5,25),
                  "weights":['uniform', 'distance'],
                  "algorithm":['auto', 'ball_tree', 'kd_tree', 'brute']
                  }

knn_grid = GridSearchCV(KNN_clfr, KNN_parameters)
knn_grid.fit(x_train,y_train)

knn_grid.best_params_

best_model = knn_grid.best_estimator_
best_model

KNNh_clf = best_model

KNNh_clf.fit(x_train, y_train)

KNNh_predictions = KNNh_clf.predict(x_test)

KNNh_conf_mat = metrics.confusion_matrix(y_test,KNNh_predictions)

KNNh_conf_mat

axis_ticks=[0,1] # name  of classes
fig, ax = plt.subplots()
ticks = np.arange(len(axis_ticks))
plt.xticks(ticks, axis_ticks)
plt.yticks(ticks, axis_ticks)

#im = ax.imshow(confusion_mat, cmap="Pastel1", aspect="auto")
sns.heatmap(pd.DataFrame(KNNh_conf_mat), annot=True, cmap="Pastel1")
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('-----!! Confusion Matrix for Heart Disease' 
          '- K-Nearest Neigbours (Hypertuned)!!-----', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

ht_KNN_acc_score = KNNh_clf.score(x_test,y_test)
ht_KNN_acc_score

#%%

#K-Means

kx = x[["Age","RestingBP","Cholesterol","MaxHR"]]

ky = y.copy()

kx_train,kx_test,ky_train,ky_test=train_test_split(kx,ky,test_size=0.2)

ksc = StandardScaler()
kx_train = ksc.fit_transform(kx_train)
kx_test = ksc.transform(kx_test)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
#  the sum of squared distance between each point and the centroid in a cluster
wcss = []
for i in range(1, 11):
    
    model_kmeans = KMeans(n_clusters = i, init = 'k-means++',
                          random_state = 42)
    
    model_kmeans.fit(kx_train)
    
    wcss.append(model_kmeans.inertia_)
    
plt.subplots()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

KM_clfr = KMeans(n_clusters=2)

KM_parameters = {"random_state":np.arange(10,50,10)}

grid = GridSearchCV(KM_clfr, KM_parameters)
grid.fit(kx_train,ky_train)

grid.best_params_

best_model =grid.best_estimator_
best_model
# Fitting K-Means to the dataset
model_kmeans = best_model
# fit
model_kmeans.fit(kx_train)
# Predict
ky_kmeans = model_kmeans.predict(kx_train)

# or fit and predict at once
#y_kmeans = kmeans.fit_predict(x_train)

# take a look at the comparison between y and y_pred
y_train_com = np.append(ky_train.values.reshape(ky_train.shape[0],1),\
                    ky_kmeans.reshape(ky_kmeans.shape[0],1), axis = 1)

feat1 = 0
feat2 = 3

#-- The actual
plt.subplots()
plt.scatter(kx_train[ky_train == 1, feat1], kx_train[ky_train == 1, feat2],
            s = 20, c = 'red', label = 'Cluster 1')
plt.scatter(kx_train[ky_train == 2, feat1], kx_train[ky_train == 2, feat2],
            s = 20, c = 'blue', label = 'Cluster 2')
plt.scatter(model_kmeans.cluster_centers_[:, feat1],
            model_kmeans.cluster_centers_[:, feat2],\
            s = 20, c = 'yellow', label = 'Centroids')

plt.title('Clusters of types')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#-- The predictions
plt.subplots()
plt.scatter(kx_train[ky_kmeans == 0, feat1], kx_train[ky_kmeans == 0, feat2],
            s = 20, c = 'red', label = 'Cluster 1')
plt.scatter(kx_train[ky_kmeans == 1, feat1], kx_train[ky_kmeans == 1, feat2],
            s = 20,  c = 'blue', label = 'Cluster 2')
plt.scatter(model_kmeans.cluster_centers_[:, feat1],
            model_kmeans.cluster_centers_[:, feat2],
            s = 20, c = 'yellow', label = 'Centroids')
plt.title('Clusters of types')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#%%

print("Logistic Regression: \n",
      log_confusion_mat,"\nAccuracy:", log_reg_acc_score, "\n\n",
      "Logistic Regression with Hypertuning: \n\n",
      ht_log_confusion_mat,"\nAccuracy:", ht_log_reg_acc_score, "\n\n",
      "Decision Tree: \n",
      DT_conf_mat,"\nAccuracy:", dec_tree_acc_score, "\n\n",
      "Decision Tree with Hypertuning: \n\n",
      DTh_conf_mat,"\nAccuracy:", ht_dec_tree_acc_score, "\n\n",
      "Random Forest: \n",
      RF_conf_mat,"\nAccuracy:", ran_for_acc_score, "\n\n",
      "Random Forest with HyperTuning: \n\n",
      RFh_conf_mat,"\nAccuracy:", ht_ran_for_acc_score, "\n\n",
      "SVC: \n",
      SVC_conf_mat,"\nAccuracy:", SVC_acc_score, "\n\n",
      "SVC with Hypertuning: \n\n",
      SVCh_conf_mat,"\nAccuracy:", ht_SVC_acc_score, "\n\n",
      "K-Nearest Neighbours: \n",
      KNN_conf_mat,"\nAccuracy:", KNN_acc_score, "\n\n",
      "K-Nearest Neighbours with Hypertuning: \n\n",
      KNNh_conf_mat,"\nAccuracy:", ht_KNN_acc_score)
