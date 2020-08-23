from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB,BernoulliNB

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, Perceptron, PassiveAggressiveClassifier

from sklearn import preprocessing

from tqdm import tqdm
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold


def run_model(X_train, y_train, X_test, y_test):
    y_pred_all=[]
    accuracy_all = []
    clf = linear_model.SGDClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Testing accuracy  SGDClassifier(SVM):  %s' % (accuracy_score(y_test, y_pred) * 100))
    accuracy_all.append(accuracy_score(y_test, y_pred) * 100)
    # accuracy[i][0]=(accuracy_score(y_test, y_pred)*100)
    # a = []
    # a.append(accuracy_score(y_test, y_pred) * 100)
    y_pred_all.append(y_pred)
    clf = linear_model.SGDClassifier(n_jobs=-1, loss="log")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Testing accuracy SGDClassifier(Logistic regression) : %s' % (accuracy_score(y_test, y_pred) * 100))
    accuracy_all.append(accuracy_score(y_test, y_pred) * 100)
    y_pred_all.append(y_pred)

    clf = Perceptron(n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Testing accuracy Perceptron %s' % (accuracy_score(y_test, y_pred) * 100))
    accuracy_all.append(accuracy_score(y_test, y_pred) * 100)
    # accuracy[i][1]=(accuracy_score(y_test, y_pred)*100)
    # a.append(accuracy_score(y_test, y_pred) * 100)
    y_pred_all.append(y_pred)

    clf = PassiveAggressiveClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_all.append(y_pred)
    accuracy_all.append(accuracy_score(y_test, y_pred) * 100)
    print('Testing accuracy PassiveAggressiveClassifier %s' % (accuracy_score(y_test, y_pred) * 100))
    # a.append(accuracy_score(y_test, y_pred) * 100)
    # accuracy[i][2]=(accuracy_score(y_test, y_pred)*100)
    clf= KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_all.append(y_pred)
    accuracy_all.append(accuracy_score(y_test, y_pred) * 100)
    print('Testing accuracy KneighborClassifier %s' % accuracy_score(y_test, y_pred))
    # clf= MLPRegressor()
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print('Testing accuracy svm %s' % accuracy_score(y_test, y_pred))

    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_all.append(accuracy_score(y_test, y_pred) * 100)
    print('Testing accuracy BernoulliNB %s' % (accuracy_score(y_test, y_pred) * 100))
    # a.append(accuracy_score(y_test, y_pred) * 100)
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_all.append(accuracy_score(y_test, y_pred) * 100)
    print('Testing accuracy DecisionTreeClassifier %s' % (accuracy_score(y_test, y_pred) * 100))

    clf = RandomForestClassifier(criterion='entropy',n_jobs=-1,max_depth=50,max_features='auto',n_estimators=300)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_all.append(accuracy_score(y_test, y_pred) * 100)
    print('Testing accuracy RandomForestClassifier %s' % (accuracy_score(y_test, y_pred) * 100))

    #this comment return is for KFOLD
    # return y_pred_all,accuracy_all
    return y_pred_all


# Program to find most frequent
# element in a list

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num

df = pd.read_csv('hotel_bookings.csv')



# df=df[: 10000]

# print(df.isnull().sum())
df['country'].fillna("unknown", inplace=True)
# print(df.isnull().sum())
df.fillna(0, inplace=True)
# print(df['country'].unique())



le = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()

le.fit(df['country'])
le2.fit(df['hotel'])
df['country']=le.transform(df['country'].values)
df['hotel']=le2.transform(df['hotel'].values)

df['is_repeated_guest'] = le.fit_transform(df['is_repeated_guest'])
df['reserved_room_type'] = le.fit_transform(df['reserved_room_type'])
df['assigned_room_type'] = le.fit_transform(df['assigned_room_type'])
df['deposit_type'] = le.fit_transform(df['deposit_type'])
df['agent'] = le.fit_transform(df['agent'])

df['arrival_date_day_of_month'] = le.fit_transform(df['arrival_date_day_of_month'])
df['customer_type'] = le.fit_transform(df['customer_type'])
df['meal'] = le.fit_transform(df['meal'])
df['market_segment'] = le.fit_transform(df['market_segment'])
df['distribution_channel'] = le.fit_transform(df['distribution_channel'])
df['arrival_date_month'] = le.fit_transform(df['arrival_date_month'])

# x1=df.loc[: , "days_in_waiting_list":"days_in_waiting_list"]

# df['avg'] = (df['total_of_special_requests'] + df['required_car_parking_spaces']+df['days_in_waiting_list']+df['booking_changes'])/4
# df.drop(['is_canceled'], axis=1)
# df.drop(['hotel'], axis=1)
df1 = df[['arrival_date_day_of_month','arrival_date_month','stays_in_weekend_nights','previous_bookings_not_canceled','days_in_waiting_list','distribution_channel','booking_changes','market_segment','meal','customer_type','agent',
'deposit_type','assigned_room_type','reserved_room_type','hotel','lead_time','stays_in_week_nights','previous_cancellations',
          'is_repeated_guest','total_of_special_requests', 'required_car_parking_spaces','country','adults','lead_time','is_canceled']]

# df1 = df[['arrival_date_month','distribution_channel','market_segment','meal','customer_type','agent',
# 'deposit_type','assigned_room_type','required_car_parking_spaces','country','adults','lead_time','is_canceled']]


# print(df['agent'].unique())
train, test = train_test_split(df1, test_size=0.3, random_state=7)


x1 = train.drop(['is_canceled'], axis = 1).values
x2 = test.drop(['is_canceled'], axis = 1).values

#
# run_model(train.values.reshape(-1, 1), train['is_canceled'].values.reshape(-1, 1),
#            test.values.reshape(-1, 1), test['is_canceled'].values.reshape(-1, 1))

x3=run_model(x1, train['is_canceled'].values,
           x2, test['is_canceled'].values)




#code zir baraye ghesmat KFOLD ast
# kf=KFold(n_splits=2, random_state=7, shuffle=True)
# acc_final=[]
# for train_index, test_index in kf.split(df1):
#     x1 = df1.drop(['is_canceled'], axis=1).values
#     x2 = df1.drop(['is_canceled'], axis=1).values
#     x3,x4 = run_model(x1[train_index], df1['is_canceled'][train_index].values,
#                    x2[test_index], df1['is_canceled'][test_index].values)
#     acc_final.append((x4))
# numpy_array=np.array(acc_final)
# # for i in range(10):
# #     print(acc_final[i][0])
# c = numpy_array.mean(axis=0)
# print(c)







# code zire baraye ravashe ensemble(raye aksariyat) ast
y_pred_final=[]
for i in tqdm(range(len(x3[0]))):
   a = []
   for j in range(len(x3)):
       a.append(x3[j][i])
#
   y_pred_final.append(most_frequent(a))

# print(y_pred_final)
accuracy = accuracy_score(test['is_canceled'].values.reshape(-1, 1), y_pred_final) * 100
print("Testing accuracy of ensemble is %s:" % accuracy)

