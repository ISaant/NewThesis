import matplotlib.pyplot as plt
import numpy as np

from Generate_Features_Dataloades import Generate
from importlib import reload
from Fun4newThesis import *
from FunClassifiers4newThesis_pytorch import *
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

psd2use, restStatePCA, anat2use, anatPCA, DiagFc, restStatePCA_s200, anatPCA_s200, local, glob, ROIs, scores = Generate()

#%% Prepare dataset

print('Generating Dataset...')
# scores_scaled,scaler=Scale_out(scores)
dataset=CustomDataset(restStatePCA, anatPCA, DiagFc, restStatePCA_s200, anatPCA_s200, np.concatenate((np.array(local), glob),axis=1), scores, transform=None)


#%%

def dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=[]):
    gen = torch.Generator()
    if seed:
        gen.manual_seed(seed)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size], generator = gen)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)

    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset),
                             shuffle=False, num_workers=2)

    val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset),
                            shuffle=False, num_workers=2)

    return train_loader, test_loader, val_loader

dataloader=DataLoader(dataset=dataset,batch_size=200,shuffle=True,num_workers=2)
dataiter=iter(dataloader)
data=next(dataiter)
#----------------------#
#For indexing dataset/dataiter: 0:psd,1:anat,2:embeddings,3:tagets
#----------------------#

psd,anat, fc, psds200, anats200, ScFeat, l= data
print(psd.shape ,anat.shape , fc.shape, psds200.shape, anats200.shape, ScFeat.shape, l.shape)

# Define input/output sizes, batchsize and learning rate

train_size = int(0.7 * len(dataset))
test_size = int(0.9 * len(dataset)) - train_size
val_size = len(dataset) - train_size - test_size

print(train_size+test_size+val_size == len(dataset))

batch_size=128 #Asegurate de que sea mayor al tamaño del test
lr = .0001
num_epochs = 150
input_size_psd = psd.shape[1]
input_size_anat = anat.shape[1]
input_size_fc = fc.shape[1]
input_size_psds200 = psds200.shape[1]
input_size_anats200 = anats200.shape[1]
input_size_ScFeat = ScFeat.shape[1]
input_size_global_sc = glob_sc.shape[1]
output_size = scores.shape[1]


#%%
iterations = 3
model = []
import FunClassifiers4newThesis_pytorch
reload(FunClassifiers4newThesis_pytorch)
from FunClassifiers4newThesis_pytorch import *

#%% Lasso For Anat
#


Mean_Acc=[]
ROIcsv=pd.read_csv('~/Documents/Doctorado_CIC/NewThesis/dk_ROI_Importance.csv')
for i in range(50):
    anat_train, anat_test, y_train, y_test, _, _ = Split(anatPCA, scores, .3)
    anat_train = Scale(anat_train)
    anat_test = Scale(anat_test)
    model = Lasso(alpha=.2)
    model.fit(anat_train, y_train)
    pred_Lasso = model.predict(anat_test)
    LassoPred = plotPredictionsReg(pred_Lasso, y_test.flatten(), False)
    Mean_Acc.append(LassoPred)
Mean_Acc=np.mean(Mean_Acc)

anatPCA = myReshape(anatPCA)
Sub,Feat,ROI = anatPCA.shape
ROI_Importance = []
for roi in tqdm(range(ROI)):
    reg_roi=[]
    for i in range(50):
        anat_train, anat_test, y_train,y_test,_,_=Split(anatPCA[:,:,roi], scores,.3)
        anat_train = Scale(anat_train)
        anat_test = Scale(anat_test)
        model = Lasso(alpha=.2)
        model.fit(anat_train, y_train)
        pred_Lasso = model.predict(anat_test)
        LassoPred = plotPredictionsReg(pred_Lasso, y_test.flatten(), False)
        reg_roi.append(LassoPred)
    ROI_Importance.append(np.mean(reg_roi))
ROI_Importance = np.take(ROI_Importance,np.argsort(ROIs))
# ROI_Importance = Mean_Acc*(1-(Mean_Acc-ROI_Importance))

ROIcsv['Acc'] = ROI_Importance
ROIcsv.to_csv('~/Documents/Doctorado_CIC/NewThesis/dk_ROI_Importance.csv',index=False)

sns.barplot(ROIcsv,x='network', y = 'Acc', palette='mako')
plt.xticks(rotation = 45)
plt.ylim(.3,.71)
anatPCA = RestoreShape(anatPCA)
#%% Individual NN
#psd
NNPred_psd=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = NeuralNet(input_size_psd, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    # Puedes imprimir el resumen del modelo si lo deseas
    # print(model)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = 0
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    # pred = scaler.inverse_transform(np.array(pred).reshape(-1,1))
    # y_test = scaler.inverse_transform(test_data[:][-1].numpy())
    # NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_psd.append(NNPred)
#%%anat
NNPred_anat=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = NeuralNet(input_size_anat, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    # Puedes imprimir el resumen del modelo si lo deseas
    # print(model)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = 1
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    # pred = scaler.inverse_transform(np.array(pred).reshape(-1,1))
    # y_test = scaler.inverse_transform(test_data[:][-1].numpy())
    # NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_anat.append(NNPred)

#%%psd_s200
NNPred_psd_s200=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = NeuralNet(input_size_psds200, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    # Puedes imprimir el resumen del modelo si lo deseas
    # print(model)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = 3
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    # pred = scaler.inverse_transform(np.array(pred).reshape(-1,1))
    # y_test = scaler.inverse_transform(test_data[:][-1].numpy())
    # NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_psd_s200.append(NNPred)
#%%anat_s200
NNPred_anat_s200=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = NeuralNet(input_size_anats200, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    # Puedes imprimir el resumen del modelo si lo deseas
    # print(model)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = 4
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    # pred = scaler.inverse_transform(np.array(pred).reshape(-1,1))
    # y_test = scaler.inverse_transform(test_data[:][-1].numpy())
    # NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_anat_s200.append(NNPred)
#%% Parallel NN
NNPred_CustomModel=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = CustomModel_NoFc(input_size_psd, input_size_anat, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = [0, 1]
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_CustomModel.append(NNPred)
#%%
NNPred_CustomModel_s200=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = CustomModel_NoFc(input_size_psds200, input_size_anats200, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = [3, 4]
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_CustomModel_s200.append(NNPred)

#%%
NNPred_CustomModel_Sc=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = CustomModel_Sc(input_size_psds200, input_size_anats200, input_size_ScFeat,  output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = [3, 4, 5]
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_CustomModel_Sc.append(NNPred)
    NNPred_CustomModel_Sc2.append(NNPred)

#%%
PredExperimentsDF=pd.DataFrame({'PSD_PCA':list(np.array(NNPred_psd_s200)),
                                'Anat_PCA':list(np.array(NNPred_anat_s200)),
                                'Parallel_noSc':list(np.array(NNPred_CustomModel_s200)),
                                'Parallel_Sc':list(np.array(NNPred_CustomModel_Sc)),
                                #'GCN': models_acc[0],
                                #'SAGE_GCN': models_acc[1],
                                #'GNN_Diffpool': models_acc[2],
                                #'GIN':NNPred_list_CustomModel_Fc
                                })

plt.figure()
PredExperimentsDf_melted = PredExperimentsDF.reset_index().melt(id_vars='index')
sns.boxplot(PredExperimentsDf_melted,y='value',x='variable', palette="dark:#5A9_r")
plt.xlabel('Model')
plt.ylabel('Performance')
plt.title('Boxplot Acc')

#%%

