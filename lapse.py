import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


conn=sqlite3.connect(':memory:')
print("connection connected:",conn)

def sql(query):
    return pd.read_sql_query(query,conn)

df=pd.read_csv("data/clean/orders.csv")
df['order_ts']=pd.to_datetime(df['order_ts'])
cutoff=pd.Timestamp('2025-09-01')

completed=df[df['status']=='completed']

pc_completed=completed[completed['order_ts']<cutoff] #pre cutoff completed
cohort_ids=pc_completed['customer_id'].unique()

sep_ids = set(completed[(completed['order_ts'] >= '2025-09-01') & (completed['order_ts'] < '2025-10-01')]['customer_id'])

labels = pd.DataFrame({'customer_id': pc_completed['customer_id'].unique()})
labels['lapsed'] = (~labels['customer_id'].isin(sep_ids)).astype(int)

print("Cohort size:", len(labels))
print(labels['lapsed'].value_counts())

feat=pc_completed.groupby('customer_id').agg(
    last_completed=('order_ts', 'max'),
    frequency=('order_id', 'nunique'),
    monetary=('order_amount', 'sum'),
    avg_discount_pct=('discount_pct', 'mean'),
    avg_delivery_minutes=('delivery_minutes', 'mean'),
).reset_index()
feat['recency']=(cutoff-feat['last_completed']).dt.days
feat=feat.drop(columns='last_completed')

final=completed[completed['status'].isin(['completed','cancelled'])]
cancel=final.groupby('customer_id').agg(
    total_final=('order_id', 'count'),
    cancelled=('status', lambda s: (s=='cancelled').sum()),
).reset_index()
cancel['cancellation_rate']=cancel['cancelled']/cancel['total_final']

labels=labels.merge(feat,on='customer_id',how='left')
labels=labels.merge(cancel[['customer_id','cancellation_rate']],on='customer_id',how='left')

print(labels.head())

dfc=pd.read_csv('data/clean/customers.csv')

dfc['signup_date']=pd.to_datetime(dfc['signup_date'])
dfc['tenure_days']=(cutoff - dfc['signup_date']).dt.days

labels.to_sql('ocohort',conn,index=False,if_exists='replace')
dfc.to_sql('customers',conn,index=False,if_exists='replace')

query="""
SELECT * FROM ocohort o
INNER JOIN customers c ON o.customer_id=c.customer_id
"""

clabels=sql(query)

clabels=clabels.loc[:, ~clabels.columns.duplicated()]
clabels=clabels.drop(columns='signup_date')
print(clabels["lapsed"].value_counts())
print("number of customer:",len(clabels))

num_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))

cols=clabels.columns.tolist()
laps=cols.pop(1)
cols.pop(0)
x=clabels[cols]
y=clabels[laps]

num_cols = ['recency','frequency','monetary','avg_discount_pct','avg_delivery_minutes','cancellation_rate','age','tenure_days']
cat_cols = ['city','gender']

prep=ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols),
])

model=make_pipeline(prep, LogisticRegression(max_iter=1000, class_weight='balanced'))

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,     
    random_state=42,     
    stratify=y  # keeps the ratio the same in both train and test
)

print("Train rows:",len(y_train))
print("Test rows:",len(y_test))

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred),4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0),4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0),4))
print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0),4))



baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(x_train, y_train)
y_base = baseline.predict(x_test)

print("\nBaseline accuracy:", accuracy_score(y_test, y_base))
print("Baseline lapsed F1:", f1_score(y_test, y_base, pos_label=1, zero_division=0))

print("\n",y_test.value_counts(normalize=True)*100)




dfp=pd.read_csv('data/clean/products.csv')

pre_completed=df[(df['order_ts'] < cutoff) & (df['status'] == 'completed')]

cat_orders=pre_completed.merge(dfp[['product_id', 'category']], on='product_id', how='left')

top_cat = (cat_orders.groupby(['customer_id', 'category']).size().reset_index(name='n').sort_values(['customer_id', 'n', 'category'], ascending=[True, False, True])
           .drop_duplicates('customer_id')[['customer_id', 'category']].rename(columns={'category': 'top_category'}))

cplabels = clabels.merge(top_cat, on='customer_id', how='left')
assert len(cplabels) == 1156

num_cols = ['recency','frequency','monetary','avg_discount_pct','avg_delivery_minutes','cancellation_rate','age','tenure_days']
cat_cols = ['city','gender','top_category']

dfpre=pd.read_csv("outputs/customer_segments.csv") #creating premium label table
dfpre.to_sql("customer_segments",conn,index=False,if_exists="replace")
cplabels.to_sql("cplabels",conn,index=False,if_exists="replace")

query="""
SELECT * FROM customer_segments c
INNER JOIN cplabels l ON c.customer_id=l.customer_id
"""
prelabels=sql(query)
conn.close()

cplabels=cplabels.drop(columns=["customer_id"])
cols=cplabels.columns.tolist()
laps=cols.pop(0)
x=cplabels[cols]
y=cplabels[laps]

prep=ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols),
])

model=make_pipeline(prep, LogisticRegression(max_iter=1000, class_weight='balanced'))

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,     
    random_state=42,     
    stratify=y  # keeps the ratio the same in both train and test
)

a=cplabels["frequency"].mean()*100
b=x_train["frequency"].mean()*100
print("mean of frequency of full data:",round(a,3))
print("mean of frequency of train data:", round(b,3))

param_grid={"logisticregression__C": [0.01, 0.1, 1, 10, 100]}

grid_search=GridSearchCV(model, param_grid, cv=5, scoring="f1")
grid_search.fit(x_train, y_train)

print("\nBest C found:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
test_f1=best_model.score(x_test, y_test)
print("Final test f1 (on the untouched test set):", test_f1)


print("\nPrecision of Logistic regression:", round(precision_score(y_test, y_pred, zero_division=0),4))
print("Recall of Logistic Regression:", round(recall_score(y_test, y_pred, zero_division=0),4))
print("F1 Score of Logistic regression:", round(f1_score(y_test, y_pred, zero_division=0),4))

rf_model = make_pipeline(
    prep,   # the same ColumnTransformer used for logistic regression
    RandomForestClassifier(class_weight='balanced', random_state=42)
)

rf_param_grid = {
    'randomforestclassifier__n_estimators': [100],
    'randomforestclassifier__max_depth': [3, 5, 8],
    "randomforestclassifier__min_samples_leaf": [5, 10, 20],
    "randomforestclassifier__max_features": ["sqrt", 0.5]
}

rf_grid_search=GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='f1')
rf_grid_search.fit(x_train, y_train)  

print("\n\nGrid search completed.")
print("Best parameters:")
print(rf_grid_search.best_params_)
print("\nBest mean CV F1:",round(rf_grid_search.best_score_, 4))

best_model=rf_grid_search.best_estimator_   
y_pred=best_model.predict(x_test)

test_f1 = f1_score(y_test, y_pred, pos_label=1)
print("\nTest F1 (lapsed):", round(test_f1, 4))
print("Meets target:", test_f1 >= 0.55)



smote_model = ImbPipeline([
    ('columntransformer', prep),          # imputation + scaling + one-hot encoding
    ('smote', SMOTE(random_state=42)),
    ('logisticregression', LogisticRegression(max_iter=1000)),
])

smote_model.fit(x_train, y_train)
y_pred = smote_model.predict(x_test)

grid_search_smote=GridSearchCV(
    smote_model,
    {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',
)
grid_search_smote.fit(x_train, y_train)

print("\nPrecision of balanced labels:", round(precision_score(y_test, y_pred, zero_division=0),4))
print("Recall of balanced labels:", round(recall_score(y_test, y_pred, zero_division=0),4))
print("F1 Score of balanced labels:", round(f1_score(y_test, y_pred, zero_division=0),4))



rf_best = rf_grid_search.best_estimator_
rf_clf = rf_best.named_steps['randomforestclassifier']

feature_names = rf_best.named_steps['columntransformer'].get_feature_names_out()

importance = pd.Series(rf_clf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\n\nTop 5 feature importance of Random Forrest\n",importance.head())

best_lr = grid_search.best_estimator_
ct = best_lr.named_steps['columntransformer']
lr = best_lr.named_steps['logisticregression']

feature_names = ct.get_feature_names_out()
coefs = pd.Series(lr.coef_[0], index=feature_names).sort_values()
print("\n",coefs.head(3),"\n\nTop feature is Frequency which shows that when it decreases, lapse increases\n" \
"Second top feature is Electronics category which shows when more people buy electronic products, lapse decreases\n" \
"Third top feature is Delhi city which shows that people in Delhi has high lapse")


joblib.dump(grid_search.best_estimator_, "outputs/lapse_model.joblib") #Logistic regression model

model=joblib.load("outputs/lapse_model.joblib")

y_pred=model.predict(x_test)

print("\nPrecision of loaded model:", round(precision_score(y_test, y_pred, zero_division=0),4))
print("Recall of loaded model:", round(recall_score(y_test, y_pred, zero_division=0),4))
print("F1 Score of loaded model:", round(f1_score(y_test, y_pred, zero_division=0),4))




cols=prelabels.columns.tolist()
lapse=cols.pop(6)
prelabels=prelabels.loc[:, ~prelabels.columns.duplicated()]
x=prelabels[cols]
y=prelabels[lapse]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,     
    random_state=42,     
    stratify=y  # keeps the ratio the same in both train and test
)

best=grid_search.best_estimator_

x_test=x_test.loc[:, ~x_test.columns.duplicated()]

results = pd.DataFrame({
    'customer_id': prelabels.loc[x_test.index, 'customer_id'].values,
    'lapse_probability': best.predict_proba(x_test)[:, 1],
    'actually_lapsed': best.predict(x_test)
}, index=x_test.index)

results=results.sort_values('lapse_probability', ascending=False)
rlist=results.head(10)

rlist.to_csv("outputs/retention_list.csv",index=False)
