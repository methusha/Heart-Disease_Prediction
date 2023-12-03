# -*- coding: utf-8 -*-
"""

@author: methusha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import pickle

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
print(ran_for_acc_score)

#%%

# Example usage
input_features = [[40,1,1,140,289,0,1,172,0,0,2], [49,0,2,160,180,0,1,156,0,1,1],
                  [37,1,1,130,283,0,2,98,0,0,2], [60,1,2,160,300,0,1,125,0,1,1]]

test_feature = [[44,1,1,150,288,0,1,150,1,3,1]]
prediction = RF_clf.predict(test_feature)
print(prediction)

#%%

# Save the model and label encoder using pickle
data = {"model": RF_clf, "label_encoder": label_encoder}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)

# Load the model and label encoder
with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

RF_loaded = data["model"]
label_encoder_loaded = data["label_encoder"]


# Make predictions
new_prediction = RF_loaded.predict(test_feature)
print(new_prediction)













































#%%

