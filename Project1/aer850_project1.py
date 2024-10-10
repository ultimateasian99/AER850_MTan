## Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#ML algorithm tings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,f1_score,confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib


## STEP 1: DATA PROCESSING
df = pd.read_csv("Project_1_Data.csv")

# data separation as x and y
y = df.iloc[:,3]
X = df.drop('Step',axis = 1)

## STEP 2: DATA VISUALIZATION 
fig1 = plt.figure()
ax = plt.axes(projection ='3d')
scatter = ax.scatter(df['X'],df['Y'],df['Z'], c=df['Step'], cmap='plasma')

# # add color scale
color_bar = plt.colorbar(scatter)
color_bar.set_label('Step')

# #label axes and title
plt.title('3D coordinates vs Step')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.tight_layout()
plt.show()

fig3 = plt.figure()
plt.hist(y)
plt.xlabel("Steps")
plt.ylabel("Frequency")
plt.title("Distribution of Step")
plt.show()


# ## STEP 3: CORRELATION ANALYSIS
corr = df.corr(method = 'pearson')
print(corr)

# # heat map of correlation
fig2 = plt.figure()
sns.heatmap(corr, cmap='coolwarm',vmin=-1, vmax=1,annot = True)     # heat map of the correlation matrix
plt.title('Correlation Heat Map')
plt.show()


## STEP 4: CLASSIFICATION MODEL DEVELOPMENT/ENGINEERING

#splitting the data into test and train set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.3, random_state=43,stratify=y)


# Random Forest model initialization , via SearchGridCV
RF = RandomForestClassifier(random_state=42)

# #Set grid of parameters
RF_params = [{
        'n_estimators': [10, 100, 1000],
        'criterion': ['entropy','gini'],
        'min_samples_leaf': [1,2,4,10],
        'max_depth': [10,20,30,None]
}]

# # Initalize gridsearchCV model
grid_RF = GridSearchCV(RF,
                       RF_params,
                       cv = 5,
                       scoring='accuracy',
                       n_jobs=-1
)
# Train model
grid_RF.fit(X_train,y_train)
best_RF_model = grid_RF.best_estimator_

print("Best RF Estimator:",best_RF_model)
print("Best RF Score:",grid_RF.best_score_)

# # Performance analysis of RF model, step 5
RF_ypred = best_RF_model.predict(X_test)
RF_f1 = f1_score(y_test,RF_ypred, average='weighted')
RF_accuracy = accuracy_score(y_test,RF_ypred)
RF_precison = precision_score(y_test,RF_ypred, average='weighted')
print("RF Model:\nF1 Score:",RF_f1,"\nAccuracy Score:",RF_accuracy,"\n Precision Score:",RF_precison)
print(classification_report(y_test, RF_ypred))

# # Confusion Matrix
RF_confusion = confusion_matrix(y_test,RF_ypred, labels=grid_RF.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=RF_confusion,display_labels=grid_RF.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for RF')
plt.show()


# SVC  model
SV = SVC()

#Set grid of parameters
SV_params = [{
    'kernel': ['linear','poly','rbf'],
    'C': [0.1,1,10,20],
    'degree': [2,3,4,5],
    'gamma': ['auto',0.1,1],
    'class_weight': [None, 'balanced']
}]

grid_SV = GridSearchCV(
    SV,
    SV_params,
    cv=5,
    scoring = 'accuracy',
    n_jobs=-1
)

grid_SV.fit(X_train,y_train)
best_SV_model = grid_SV.best_estimator_

print("\nBest SVC Estimator:",best_SV_model)
print("Best SVC Score:",grid_SV.best_score_)

# Performance analysis of the SV model, step 5
SV_ypred = best_SV_model.predict(X_test)
SV_f1 = f1_score(y_test,SV_ypred, average='weighted')
SV_accuracy = accuracy_score(y_test,SV_ypred)
SV_precison = precision_score(y_test,SV_ypred, average='weighted')
print("SV Model:\nF1 Score:",SV_f1,"\nAccuracy Score:",SV_accuracy,"\nPrecision Score:",SV_precison)
 
# Confusion Matrix
SV_confusion = confusion_matrix(y_test,SV_ypred, labels=grid_SV.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=SV_confusion,display_labels=grid_SV.classes_)
disp.plot(cmap='Blues')
plt.title ('Confusion Matrix for SVC')
plt.show()

# Decision Tree model
DT = DecisionTreeClassifier()

# Set grid of parameters
DT_params=[{
    'max_depth': range(2,10,2),
    'min_samples_split': range(5,50,10),
    'min_samples_leaf': range(5,50,10),
    'criterion':['gini','entropy','log_loss']
}]

grid_DT = GridSearchCV(
    DT,
    DT_params,
    cv = 5,
    scoring = "accuracy",
    n_jobs=-1
)

# # Train the model
grid_DT.fit(X_train,y_train)

best_DT_model = grid_DT.best_estimator_

print("\nBest DT Estimator:",best_DT_model)
print("Best DT Score:",grid_DT.best_score_)

# # Performance analysis of the DT model, step 5
DT_ypred = best_DT_model.predict(X_test)
DT_f1 = f1_score(y_test,DT_ypred, average='weighted')
DT_accuracy = accuracy_score(y_test,DT_ypred)
DT_precison = precision_score(y_test,DT_ypred, average='weighted')

print("DT Model:\nF1 Score:",DT_f1,"\nAccuracy Score:",DT_accuracy,"\nPrecision Score:",DT_precison)

# Confusion Matrix
DT_confusion = confusion_matrix(y_test,DT_ypred, labels=grid_DT.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=DT_confusion,display_labels=grid_DT.classes_)
disp.plot(cmap='Blues')
plt.title ('Confusion Matrix for DT')
plt.show()


# # Gradient Boosting model wrt RandomizedSearchCV
GB = GradientBoostingClassifier()

# # Paramter grid for hypertuning
GB_params = [{
    'n_estimators': range(10,50,10),
    'loss': ['log_loss'],
    'learning_rate': [0.1,0.2,0.3,0.4,0,5],
    'max_depth': range(2,6,1),
    'min_samples_split': range(3,10,2)
}]

GB_search = RandomizedSearchCV(
    estimator = GB,
    param_distributions=GB_params,
    cv = 5,
    n_jobs=-1
)

# Train the model
GB_search.fit(X_train,y_train)
best_GB = GB_search.best_estimator_
print("\nBest GBC Estimator:",best_GB)
print("Best GBC Score:",GB_search.best_score_)

# # Evaluate the GBC model, step 5
GB_ypred = GB_search.predict(X_test)
GB_f1 = f1_score(y_test,GB_ypred, average='weighted')
GB_accuracy = accuracy_score(y_test,GB_ypred)
GB_precison = precision_score(y_test,GB_ypred, average='weighted')

print("RandomizedSearchCV GB Model:\nF1 Score:",GB_f1,"\nAccuracy Score:",GB_accuracy,"\nPrecision Score:",GB_precison)

# Confusion Matrix for GBC
GB_confusion = confusion_matrix(y_test,GB_ypred, labels=GB_search.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=GB_confusion,display_labels=GB_search.classes_)
disp.plot(cmap='Blues')
plt.title ('Confusion Matrix for GB')
plt.show()

## Step 6: Stacked Model Performance Analyis

# Define the models to be used for stacking
stacking_models = [
    ('rf', RF),
    ('dt', DT)
    ]

# #Define meta estimator
LR = LogisticRegression()

clf = StackingClassifier(
    estimators = stacking_models,
    final_estimator = LR
)

# Train the stacking classifier
clf.fit(X_train,y_train)

# Evaluate the stacking model, step 5
clf_ypred = clf.predict(X_test)
clf_f1 = f1_score(y_test,clf_ypred, average='weighted')
clf_accuracy = accuracy_score(y_test,clf_ypred)
clf_precison = precision_score(y_test,clf_ypred, average='weighted')

print("\nStackingClassifer Model:\nF1 Score:",clf_f1,"\nAccuracy Score:",clf_accuracy,"\nPrecision Score:",clf_precison)

# Confusion Matrix for stacked model
clf_confusion = confusion_matrix(y_test,clf_ypred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=clf_confusion,display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title ('Confusion Matrix for Stacked Model')
plt.show()



# Step 7: Model Evaluation

# Save the model to file
joblib.dump(clf,'best_model.joblib')

# Load the model from the file
load = joblib.load('best_model.joblib')


# Test given coordinates
data = [[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]]

# Predict the step values for each data point
predicted_step_values = load.predict(data)

# Print the prediction for each set of coordinates
for i, predicted_value in enumerate(predicted_step_values):
    print(f"Coordinates {data[i]} are in step: {predicted_value}")


