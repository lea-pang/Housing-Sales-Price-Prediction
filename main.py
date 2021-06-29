import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

test_data_file_path = 'test.csv'
train_data_file_path = 'train.csv'

test_data = pd.read_csv(test_data_file_path)
train_data = pd.read_csv(train_data_file_path)

le = preprocessing.LabelEncoder()

for col in train_data.columns:
	if train_data[col].dtype == object:
		train_data[col] = le.fit_transform(train_data[col])
	else:
		pass
for col in test_data.columns:
	if test_data[col].dtype == object:
		test_data[col] = le.fit_transform(test_data[col])
	else:
		pass

y = train_data.SalePrice

features = ['LotArea', 'Utilities', 'Neighborhood', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Fireplaces', 'PoolArea', 'SaleType', 'MoSold', 'YrSold']
X = train_data[features]

X_train,X_val,y_train,y_val = train_test_split(X, y, random_state=0)

house_model = RandomForestRegressor()
house_model.fit(X, y)
test_X = test_data[features]
house_preds = house_model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': house_preds})
output.to_csv('submission.csv', index=False)
print("Submission file was generated!")