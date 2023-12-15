import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import pandas as pd
from tqdm import tqdm
from Fun4newThesis import *
from FunClassifiers4newThesis_pytorch import *


#WEY! SUPER IMPORTANTE! ELIMINSTE AL SUJETO CC221585 PORQUE NO TIENE CARACTERISTICAS ANATOMICAS DE UNO DE LOS HEMISFERIOS
def Anat_Feat_s200(path, AnatFile_s200, scoreDf, subjects):
    cont = 0
    for file in tqdm(AnatFile_s200):
        if len(np.argwhere(subjects == file[:-9])) == 0:
            continue

        if cont == 0:
            anat2use = np.array(pd.read_csv(path + file)).T[np.newaxis,:]
        else:
            anat2use = np.concatenate((anat2use, np.array(pd.read_csv(path + file)).T[np.newaxis, :]))
        cont += 1

    Sub, _, ROI = anat2use.shape
    nPCA = 5
    anatPCA = np.zeros((Sub, nPCA, ROI))
    for roi in range(ROI):
        pca_df, pca2use, prop_varianza_acum = myPCA(anat2use[:, :, roi], False, nPCA)
        plt.plot(prop_varianza_acum[:10])
        anatPCA[:, :, roi] = np.array(pca2use)

    anat2use = RestoreShape(anat2use)
    anatPCA = RestoreShape(anatPCA)

    age = scoreDf['Edad'].to_numpy()
    Reg = []
    for i in tqdm(range(200)):
        x_train, x_test, y_train, y_test, _, _ = Split(anat2use, age, .3, seed=i)
        x_train = Scale(x_train)
        x_test = Scale(x_test)
        model = Lasso(alpha=.2)
        model.fit(x_train, y_train)
        pred_Lasso = model.predict(x_test)
        LassoPred = plotPredictionsReg(pred_Lasso, y_test, False)
        Reg.append(LassoPred)
        x_test = myReshape(x_test, 200)
        sub, ft, ROI = x_test.shape
        matPred = np.zeros((ft, ft))
        matPred[:] = np.nan

        for j in np.arange(0, ft - 1):
            for k in np.arange(j + 1, ft):
                # print(j,k)
                cp = copy(x_test)
                cp[:, [j, k], :] = cp[:, [k, j], :]
                cp = RestoreShape(cp)
                cp_pred_Lasso = model.predict(cp)
                cp_LassoPred = plotPredictionsReg(cp_pred_Lasso, y_test, False)
                matPred[j, k] = cp_LassoPred
                matPred[k, j] = cp_LassoPred

        matPred = np.nanmean((LassoPred - matPred) / LassoPred, axis=0)[np.newaxis,]
        if i == 0:
            MatPred = matPred
            continue
        MatPred = np.concatenate((MatPred, matPred))
    sns.displot(Reg, kde=True, color='olive')
    plt.xlabel('Desempeño')
    plt.ylabel('Densidad')
    plt.title('Desempeño MLP usando, características \n estructurales - Edad')
    # MatPredDf=pd.DataFrame(MatPred,columns=["NumVert" , "SurfArea" ,
    #                                             "GrayVol" , "ThickAvg",
    #                                             "ThickStd", "MeanCurv",
    #                                             "GausCurv", "FoldInd" ,
    #                                             "CurvInd" ])

    MatPredDf = pd.DataFrame(MatPred, columns=["NumVert", "AreaSup",
                                               "VolGris", "GrosorProm",
                                               "GrosorStd", "CurvProm",
                                               "CurvGauss", "IndPleg",
                                               "IndCurv"])
    MatPredDf_melted = MatPredDf.reset_index().melt(id_vars='index')
    plt.figure()
    sns.kdeplot(MatPredDf_melted, x='value', hue='variable', fill=True,
                common_norm=False, palette="rainbow", alpha=.5, linewidth=1)

    plt.title('Importancia de la características estructurales')
    plt.xlabel('Varianza explicada')
    plt.ylabel('Densidad')

    # anat2use = RestoreShape(
    #     np.delete(myReshape(anat2use,200), [0, 1, 7], axis=1))  # Remove the worst features for future testing
    # # Rearange for NN
    # cont = 0
    # Anat_aranged = np.zeros((anat2use.shape))  # !!! YA NO PUEDES RESTAURAR A (SUB,ANAT,ROI) USANDO RESTORESHAPE
    # for i in range(6):
    #     for j in np.arange(0, 6 * 200, 6):
    #         # print(i+j)
    #         Anat_aranged[:, cont] = anat2use[:, i + j]
    #         cont += 1
    # del model
    return anat2use, anatPCA


#%%

