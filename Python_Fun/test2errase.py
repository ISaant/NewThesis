#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:37:25 2023

@author: isaac
"""

from sklearn.preprocessing import MinMaxScaler
train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2)

test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset),
                         shuffle=False, num_workers=2)

val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset),
                        shuffle=False, num_workers=2)
model = NeuralNet(input_size_psd, output_size).to(device)

scaler=MinMaxScaler(feature_range=(-1,1))
scaler.fit(labels)
if 'model' in globals():
    model.apply(weight_reset)
print(model._get_name())
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
var_pos = 0
train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
mse_test, pred = test_ANN(model, test_loader, var_pos)
pred = scaler.inverse_transform(np.array(pred).reshape(-1,1))
y_test = scaler.inverse_transform(test_dataset[:][-1].numpy())
NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), True)
print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')