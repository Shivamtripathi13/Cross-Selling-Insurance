import pandas as pd
import pickle

df = pd.read_csv('new.csv')
df.head()

# from sklearn.preprocessing import LabelEncoder
# label = LabelEncoder()

# df['Gender'] = label.fit_transform(df['Gender'])

# df['Vehicle_Damage'] = label.fit_transform(df['Vehicle_Damage'])

# df['Vehicle_Age'] = df['Vehicle_Age'].replace({'< 1 Year':0, '1-2 Year':1, '> 2 Years':2})

df.drop(["id"], axis = 1,inplace=True)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X, y, test_size = 0.1, random_state=0)


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression() 
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
#print(list(zip(Y_test,Y_pred)))

print(classifier.coef_)
print(classifier.intercept_)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print("Accuracy of the model:",acc)

# from sklearn.neighbors import KNeighborsClassifier
# knc = KNeighborsClassifier(n_neighbors=5)
# knc.fit(x_train, y_train)
# y_pred = knc.predict(x_test)

# knc_score = accuracy_score(y_test,y_pred)
# print("Accuracy of the model:",knc_score)

# from sklearn.naive_bayes import GaussianNB
# NBclf = GaussianNB()
# NBclf.fit(x_train, y_train)
# y_predNB = NBclf.predict(x_test)

# NB_score = accuracy_score(y_test,y_predNB)
# print("Accuracy of the model:",NB_score)

# from sklearn.tree import DecisionTreeClassifier
# DT = DecisionTreeClassifier(random_state=0)
# DT.fit(x_train,y_train)
# y_predDT = DT.predict(x_test)

# DT_score =  accuracy_score(y_test,y_predDT)
# print("Accuracy of the model:",DT_score)

# from sklearn.ensemble import RandomForestClassifier
# RFclf = RandomForestClassifier(random_state=0)
# RFclf.fit(x_train,y_train)
# y_predRF = RFclf.predict(x_test)

# RF_score = accuracy_score(y_test, y_predRF)
# print("Accuracy of the model:",RF_score)

# from sklearn.ensemble import GradientBoostingClassifier
# GBclf = GradientBoostingClassifier(random_state=0)
# GBclf.fit(x_train, y_train)
# y_predGB = GBclf.predict(x_test)

# GB_score = accuracy_score(y_test, y_predGB)
# print("Accuracy of the model:",GB_score)

pickle.dump(classifier, open('fmodel.pkl', 'wb'))