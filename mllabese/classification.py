import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score , precision_score , recall_score , f1_score , confusion_matrix,classification_report)


data = {
    "StudyHours": [1, 2, 3, 4, 5, np.nan, 7, 8, 2, 6],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 78],
    "Gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
    "Result": ["Fail", "Fail", "Fail", "Pass", "Pass",
               "Pass", "Pass", "Pass", "Fail", "Pass"]
}

df = pd.DataFrame(data)


print("\n Original Dataset")
print(df)
print("Null Values")
print(df.isnull().sum)

#preporocessing
imputer = SimpleImputer(strategy = 'median')
df["StudyHours"] = imputer.fit_transform(df[["StudyHours"]])
encode = LabelEncoder()
df["Gender"] = encode.fit_transform(df["Gender"])
df["Result"] = encode.fit_transform(df["Result"])
print("\nDataset After Preprocessing:\n")
print(df)



#split 

X = df[["Gender","StudyHours","Attendance"]]
Y = df["Result"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.3 , random_state = 40)

models = {
    "DecisionTree" : DecisionTreeClassifier(),
    "Random Forest" : RandomForestClassifier(n_estimators = 5),
    "SVM" : SVC(kernel='rbf')
    
}

for name,model in models.items():
    print(f"{name} model")
    
    model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(Y_test , y_pred)
    precision = precision_score(Y_test , y_pred)
    recall = recall_score(Y_test , y_pred)
    f1 = f1_score(Y_test , y_pred)
    
    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    
    print("\nClassification Report:\n")
    print(classification_report(Y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(Y_test, y_pred)

    print("Confusion Matrix:\n")
    print(cm)
    
    
    plt.figure(figsize = (4,3))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.show()



