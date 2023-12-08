import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression

training_data = pd.read_csv("data/train.csv")
testing_data = pd.read_csv("data/test.csv")

print("training_data.shape[0]", training_data.shape[0])
print("testing_data.shape[0]", testing_data.shape[0])

print("training_data.head()")
print(training_data.head())
print("training_data.isnull().sum()")
print(training_data.isnull().sum())

# missing value
training_data2 = training_data[['Age', 'Sex', 'Pclass', 'Survived']].copy()
print("training_data2.head()")
print(training_data2.head())
print("training_data2.isnull().sum()")
print(training_data2.isnull().sum())

#  age
# histogram before fill in missing values
figure = plt.figure(figsize=(15,8))
plt.hist([training_data[training_data['Survived']==0]['Age'],
        training_data[training_data['Survived']==1]['Age']], 
        color = ['skyblue','peachpuff'], bins = 20,
        label = ['Not Survived','Survived'])
plt.xlabel('Age')
plt.ylabel('Population')
plt.legend()
plt.show() # .py檔要加上

mean_age = training_data['Age'].mean(skipna=True)
print("mean_age")
print(mean_age)

training_data2["Age"].fillna(mean_age, inplace=True)
print("training_data2.head()")
print(training_data2.head())

# histogram after fill in missing values
figure = plt.figure(figsize=(15,8))
plt.hist([training_data2[training_data['Survived']==0]['Age'],
        training_data2[training_data['Survived']==1]['Age']], 
        color = ['skyblue','peachpuff'], bins = 20,
        label = ['Not Survived','Survived'])
plt.xlabel('Age')
plt.ylabel('Population')
plt.legend()
plt.show() # .py檔要加上

# sex
# bar chart of sex
p_male = training_data2[training_data['Sex']=='male']['Survived'].value_counts()
p_female = training_data2[training_data['Sex']=='female']['Survived'].value_counts()
ds = pd.DataFrame([p_male,p_female])
ds.index = ['Male','Female']
ds.plot(kind='bar', figsize=(10,8), color = ['skyblue','peachpuff'],
        label = ['Not Survived','Survived'])
plt.show() # .py檔要加上

training_data2=pd.get_dummies(training_data2, columns=['Sex'])
print("training_data2.head()")
print(training_data2.head())

training_data2.drop('Sex_female', axis=1, inplace=True)
print("training_data2.head()")
print(training_data2.head())

# pclass
# bar chart of pclass
p_p1 = training_data2[training_data['Pclass']==1]['Survived'].value_counts()
p_p2 = training_data2[training_data['Pclass']==2]['Survived'].value_counts()
p_p3 = training_data2[training_data['Pclass']==3]['Survived'].value_counts()
dp = pd.DataFrame([p_p1, p_p2, p_p3])
dp.index = ['Pclass 1', 'Pclass 2', 'Pclass 3']
dp.plot(kind='bar', figsize=(10,8), color = ['skyblue','peachpuff'],
        label = ['Not Survived','Survived'])
plt.show() # .py檔要加上

# model
train_x = training_data2[['Age', 'Sex_male', 'Pclass']]
train_y = training_data2['Survived']
model = LogisticRegression(multi_class = 'multinomial')
model.fit(train_x, train_y)
model.score(train_x, train_y)

print("model.predict_proba(train_x)")
print(model.predict_proba(train_x))

# testing
testing_data2 = testing_data[['PassengerId', 'Age', 'Sex', 'Pclass']].copy()
print("testing_data2.isnull().sum()")
print(testing_data2.isnull().sum())

testing_data2["Age"].fillna(mean_age, inplace=True)
testing_data2=pd.get_dummies(testing_data2, columns=['Sex'])
testing_data2.drop('Sex_female', axis=1, inplace=True)
print("testing_data2.head()")
print(testing_data2.head())

test_x = testing_data2[['Age', 'Sex_male', 'Pclass']]
testing_pred = model.predict(test_x)

test_res = testing_data2[['PassengerId']].copy()
test_res['Survived'] = testing_pred

test_res.to_csv("data/submission.csv", index=False)