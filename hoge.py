train_data = data[data['date'] < '2022-01-01']
test_data = data[data['date'] >= '2022-01-01']
print(train_data)