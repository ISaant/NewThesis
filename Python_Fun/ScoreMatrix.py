#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 16:12:50 2023

@author: isaac
"""

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import linear_model

#%%
# os.chdir('Documents/Doctorado_CIC/Internship/Sylvain/Stability-project/Stability_project_Code/')
# os.chdir('/export03/data/Santiago/Stability_project_Code')
path2Data='/home/isaac/Documents/Doctorado_CIC/Internship/Sylvain/Stability-project/Stability-project_db/CAMCAN_Jason_PrePro/demographics_4_Guilia.csv'
path2txt='/media/isaac/Elements/materials/'
DirScores=np.sort(os.listdir(path2txt))
subjects=pd.read_csv(path2Data)['CCID'].to_numpy()
columns=['ID','Age','Sex','Acer','BentonFaces','Cattell','EmotionRecog',
         'Hotel','Ppp','Synsem','VSTM','PSD','Anat','FC']
scoreDf=pd.DataFrame(index=range(len(subjects)),columns=columns)
scoreDf['ID']=pd.read_csv(path2Data)['CCID']
scoreDf['Sex']=pd.read_csv(path2Data)['Sex']
scoreDf['Age']=pd.read_csv(path2Data)['age']
scoreDf['Acer']=pd.read_csv(path2Data)['additional_acer']

for i in np.arange(4,len(columns)):
    scoreDf[columns[i]]=np.repeat(np.nan,len(subjects))

scoreDf['PSD']=np.ones((len(subjects)))
scoreDf['Anat']=np.ones((len(subjects)))
#%% Benton Faces: Unfamiliar Faces 
# Given a target image of a face, identify same individual in an
# array of 6 face images (with possible changes in head orientation
# and lighting between target and same face in the test array)


path2benton='BentonFaces/release001/data/'
BentonFiles=np.sort(os.listdir(path2txt+path2benton))
_Pos=BentonFiles[0].find('_')+1 #find where the subject ID starts 
for files in BentonFiles:
    subPos=np.argwhere(subjects=='sub_'+files[_Pos:_Pos+8])
    if subPos.shape==(1,1) and files[-10:-4]!='failed':
        score=pd.read_csv(path2txt+path2benton+files, sep='\t')['TotalScore'].to_numpy()
        scoreDf.loc[subPos[0][0],'BentonFaces']=int(score[0])



#%% Catell: Fuild intelligence
#Complete nonverbal puzzles involving series completion,
#classiﬁcation, matrices, and conditions

path2cattell='Cattell/release001/data/'
CattellFiles=np.sort(os.listdir(path2txt+path2cattell))
_Pos=CattellFiles[0].find('_')+1 #find where the subject ID starts 
for files in CattellFiles:
    subPos=np.argwhere(subjects=='sub_'+files[_Pos:_Pos+8])
    if subPos.shape==(1,1) and files[-10:-4]!='failed':
        score=pd.read_csv(path2txt+path2cattell+files, sep='\t')['TotalScore'].to_numpy()
        scoreDf.loc[subPos[0][0],'Cattell']=int(score[0])


#%% EkmanEmHex: Emotion Expression Recognition
#View face and label emotion expressed (happy, sad, anger, fear,
#disgust, surprise) where faces are morphs along axes between
#emotional expressions.

path2emotion='EkmanEmHex/release001/data/'
EmotionRecogFiles=np.sort(os.listdir(path2txt+path2emotion))
_Pos=EmotionRecogFiles[0].find('_')+1 #find where the subject ID starts
columns=['ID','Age','Happy','Sad','Surprise','Anger','Disgust','Fear']
EmotionVsAge=pd.DataFrame(index=range(len(subjects)),columns=columns)
EmotionVsAge['ID']=pd.read_csv(path2Data)['CCID']
EmotionVsAge['Age']=pd.read_csv(path2Data)['age']
MeanCM=np.zeros((6,6)) # mean confusion matrix. 
for i in np.arange(2,len(columns)):
    EmotionVsAge[columns[i]]=np.repeat(np.nan,len(subjects))
    
cont=-1
for files in EmotionRecogFiles:
    subPos=np.argwhere(subjects=='sub_'+files[_Pos:_Pos+8])
    if subPos.shape==(1,1) and files[-10:-4]!='failed':
        score=pd.read_csv(path2txt+path2emotion+files, sep='\t')['Corr'].to_numpy()
        if not(np.isnan(score).any()):
            cm=pd.read_csv(path2txt+path2emotion+files, sep='\t')
            MeanCM+=np.array(cm.loc[:,['Ang','Dis','Fea','Hap','Sad','Sur']])
            EmotionVsAge.loc[subPos[0][0],'Anger']=int(score[0])
            EmotionVsAge.loc[subPos[0][0],'Disgust']=int(score[1])
            EmotionVsAge.loc[subPos[0][0],'Fear']=int(score[2])
            EmotionVsAge.loc[subPos[0][0],'Happy']=int(score[3])
            EmotionVsAge.loc[subPos[0][0],'Sad']=int(score[4])
            EmotionVsAge.loc[subPos[0][0],'Surprise']=int(score[5])
            cont+=1

MeanCM/=((cont)*20)
RoundAge=pd.read_csv(path2Data)['age']
RoundAge[RoundAge<30]=30
for i in np.arange(30,90,10):
    print(i)
    RoundAge[np.logical_and(RoundAge>i, RoundAge<=i+10)]=(i+10)
# RoundAge[RoundAge>80]=90
EmotionVsAge['Intervals']=RoundAge
disp =ConfusionMatrixDisplay(confusion_matrix=MeanCM,display_labels=['Ang','Dis','Fea','Hap','Sad','Sur'])
disp.plot()

df_melt=pd.melt(EmotionVsAge,id_vars=['Age'],value_vars=['Fear','Disgust','Anger','Surprise','Sad','Happy'])
color='CMRmap'
sns.displot(data=df_melt,x='value', hue='variable',kind='kde', 
            fill=True,palette=color)
sns.lmplot(x ='Age', y ='value', data = df_melt, hue ='variable', 
           scatter=False,scatter_kws={'alpha':0.3}, palette =color)

sns.residplot(data=EmotionVsAge, x="Age", y="Fear", order=1, line_kws=dict(color="r"))

def returnResuduals(df,Variables):
    
    x=np.array(copy.copy(df['Age']))
    x=x.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    resDf=copy.copy(df)
    
    for var in Variables:
        nanidx=np.array(np.where(np.isnan(df[var])))[0]
        y=np.array(df[var].fillna(df[var].mean()))
        linReg.fit(x, y)
        # Predict data of estimated models
        predictions = linReg.predict(x)
        residuals = y - predictions
        resDf[var]=residuals
        resDf.loc[nanidx,var]=np.nan
        resDf.rename(columns={var:'res'+var},inplace=True)
    return  resDf

EmotionVsAge_Residuals=returnResuduals(EmotionVsAge, ['Disgust','Anger','Fear','Happy','Sad','Surprise'])
dfRes_melt=pd.melt(EmotionVsAge_Residuals,id_vars=['Age'],value_vars=['resFear','resDisgust','resAnger','resSurprise','resSad','resHappy'])
color='mako'
sns.displot(data=dfRes_melt,x='value', hue='variable',kind='kde', 
            fill=True,palette=color)
sns.lmplot(x ='Age', y ='value', data = dfRes_melt, hue ='variable', 
           scatter=False,scatter_kws={'alpha':0.3}, palette =color)

scoreDf['EmotionRecog']=EmotionVsAge['Fear']
#%% ForceMatch: Force Matching
# Match mechanical force applied to left index ﬁnger by using right
# index ﬁnger either directly, pressing a lever which transmits force
# to left index ﬁnger, or indirectly, by moving a slider which adjusts
# the force transmitted to the left index ﬁnger.
#THERE IS ONLY 328 SUBJECTS!! DO WE WANT TO USE THIS SCORE? FOR NOW, NO

path2force='ForceMatching/release001/data/'
ForceMatchFiles=np.sort(os.listdir(path2txt+path2force))
_Pos=ForceMatchFiles[0].find('_')+1 #find where the subject ID starts 
columns=['ID','Age','FingerFinger','SlideFinger']
MatchingVsAge=pd.DataFrame(index=range(len(subjects)),columns=columns)
MatchingVsAge['ID']=pd.read_csv(path2Data)['CCID']
MatchingVsAge['Age']=pd.read_csv(path2Data)['age']
for i in np.arange(2,len(columns)):
    MatchingVsAge[columns[i]]=np.repeat(np.nan,len(subjects))
for files in ForceMatchFiles:
    subPos=np.argwhere(subjects=='sub_'+files[_Pos:_Pos+8])
    if subPos.shape==(1,1) and files[-10:-4]!='failed':
        score=np.array(pd.read_csv(path2txt+path2force+files, sep='\t')[['FingerOverCompensationMean','SliderOverCompensationMean']])[0]
        if not(np.isnan(np.array(score)).any()):
            MatchingVsAge.loc[subPos[0][0],'FingerFinger']=float(score[0])
            MatchingVsAge.loc[subPos[0][0],'SlideFinger']=float(score[1])

df_melt=pd.melt(MatchingVsAge,id_vars=['Age'],value_vars=['FingerFinger','SlideFinger'])
color='CMRmap'
sns.displot(data=df_melt,x='value', hue='variable',kind='kde', 
            fill=True,palette=color)
sns.lmplot(x ='Age', y ='value', data = df_melt, hue ='variable', 
           scatter=False,scatter_kws={'alpha':0.3}, palette =color)

# sns.residplot(data=EmotionVsAge, x="Age", y="Fear", order=1, line_kws=dict(color="r"))

# scoreDf['ForceMatch']=MatchingVsAge['some score']




#%% Hotel
#Perform tasks in role of hotel manager: write customer bills, sort
#money, proofread advert, sort playing cards, alphabetise list of
#names. Total time must be allocated equally between tasks; there is
#not enough time to complete any one task.

path2hotel='Hotel/release001/data/'
HotelFiles=np.sort(os.listdir(path2txt+path2hotel))
_Pos=HotelFiles[0].find('_')+1 #find where the subject ID starts 
columns=['ID','Age','numTask','deviation']
HotelVsAge=pd.DataFrame(index=range(len(subjects)),columns=columns)
HotelVsAge['ID']=pd.read_csv(path2Data)['CCID']
HotelVsAge['Age']=pd.read_csv(path2Data)['age']
for i in np.arange(2,len(columns)):
    HotelVsAge[columns[i]]=np.repeat(np.nan,len(subjects))
for files in HotelFiles:
    subPos=np.argwhere(subjects=='sub_'+files[_Pos:_Pos+8])
    if subPos.shape==(1,1) and files[-10:-4]!='failed':
        score=np.array(pd.read_csv(path2txt+path2hotel+files, sep='\t'))[0]
        if not(np.isnan(np.array(score)).any()):
            HotelVsAge.loc[subPos[0][0],'numTask']=float(score[0]*100)
            HotelVsAge.loc[subPos[0][0],'deviation']=float(score[1])

df_melt=pd.melt(HotelVsAge,id_vars=['Age'],value_vars=['numTask','deviation'])
color='CMRmap'
sns.displot(data=df_melt,x='value', hue='variable',kind='kde', 
            fill=True,palette=color)
sns.lmplot(x ='Age', y ='value', data = df_melt, hue ='variable', 
           scatter=True,scatter_kws={'alpha':0.3}, palette =color)

# sns.residplot(data=EmotionVsAge, x="Age", y="Fear", order=1, line_kws=dict(color="r"))

scoreDf['Hotel']=HotelVsAge['deviation']












#%% Motor Learning:
# Time-pressured movement of a cursor to a target by moving an
# (occluded) stylus under veridical, perturbed (30°), and reset
# (veridical again) mappings between visual and real space.
#THERE IS ONLY 318 SUBJECTS!! DO WE WANT TO USE THIS SCORE? FOR NOW, NO


path2motorLearning='MotorLearning/release001/data/'
MotorLearningFiles=np.sort(os.listdir(path2txt+path2motorLearning))
_Pos=MotorLearningFiles[0].find('_')+1 #find where the subject ID starts 
# for files in MotorLearningFiles:
#     subPos=np.argwhere(subjects=='sub_'+files[_Pos:_Pos+8])
#     if subPos.shape==(1,1) and files[-10:-4]!='failed':
#         score=pd.read_csv(path2txt+path2emotion+files, sep='\t')['Corr'].to_numpy()
#         if not(np.isnan(score).any()):






#%% Ppp: Picture-Picture Priming: 
# Name the pictured object presented alone (baseline), then when
# preceded by a prime object that is phonologically related (one,
# two initial phonemes), semantically related (low, high
# relatedness), or unrelated.

path2picturePriming='PicturePriming/release001/data/'
picturePrimingFiles=np.sort(os.listdir(path2txt+path2picturePriming))
_Pos=picturePrimingFiles[1].find('_')+1 #find where the subject ID starts 
columns=['ID','Age','Acc_baseline','ACC_priming','Acc_prime_high_phon',
         'Acc_prime_high_sem','Acc_prime_low_phon',
         'Acc_prime_low_sem','Acc_prime_unrelated']
ppVsAge=pd.DataFrame(index=range(len(subjects)),columns=columns)
ppVsAge['ID']=pd.read_csv(path2Data)['CCID']
ppVsAge['Age']=pd.read_csv(path2Data)['age']
for i in np.arange(2,len(columns)):
    ppVsAge[columns[i]]=np.repeat(np.nan,len(subjects))
for files in picturePrimingFiles[1:]:
    subPos=np.argwhere(subjects=='sub_'+files[_Pos:_Pos+8])
    if subPos.shape==(1,1) and files[-10:-4]!='failed':
        score=pd.read_csv(path2txt+path2picturePriming+files, sep='\t')
        ppVsAge.loc[subPos[0][0],columns[2]]=np.array(score['ACC_baseline_all'])
        ppVsAge.loc[subPos[0][0],columns[3]]=np.array(score['ACC_priming_all'])
        ppVsAge.loc[subPos[0][0],columns[4]]=np.array(score['ACC_priming_target_high_phon'])
        ppVsAge.loc[subPos[0][0],columns[5]]=np.array(score['ACC_priming_target_low_phon'])
        ppVsAge.loc[subPos[0][0],columns[6]]=np.array(score['ACC_priming_target_high_sem'])
        ppVsAge.loc[subPos[0][0],columns[7]]=np.array(score['ACC_priming_target_low_sem'])
        ppVsAge.loc[subPos[0][0],columns[8]]=np.array(score['ACC_priming_prime_unrel'])

df_melt=pd.melt(ppVsAge,id_vars=['Age'],value_vars=['Acc_baseline','ACC_priming',
                                                    'Acc_prime_high_phon','Acc_prime_high_sem',
                                                    'Acc_prime_low_phon','Acc_prime_low_sem',
                                                    'Acc_prime_unrelated'])

color='mako'
sns.displot(data=df_melt,x='value', hue='variable',kind='kde', 
            fill=True,palette=color)
sns.lmplot(x ='Age', y ='value', data = df_melt, hue ='variable', 
           scatter=False,scatter_kws={'alpha':0.3}, palette =color)

scoreDf['Ppp']=ppVsAge['Acc_baseline']

#%% Synsem: Sentence comprehension
# Listen to and judge grammatical acceptability of partial sentences,
# beginning with an (ambiguous, unambiguous) sentence stem
# (e.g., “Tom noticed that landing planes…”) followed by a disam
# biguating continuation word (e.g., “are”) in a different voice.
# Ambiguity is either semantic or syntactic, with empirically
# determined dominant and subordinate interpretations.

path2synsem1='Synsem/release001/data/'
path2synsem2='Synsem/release002/data/'
synsemFiles2=np.sort(os.listdir(path2txt+path2synsem2))
synsemFiles1=np.sort(os.listdir(path2txt+path2synsem1))
_Pos=synsemFiles1[1].find('_')+1 #find where the subject ID starts 
columns=['ID','Age','pValid','error']
synsemVsAge=pd.DataFrame(index=range(len(subjects)),columns=columns)
synsemVsAge['ID']=pd.read_csv(path2Data)['CCID']
synsemVsAge['Age']=pd.read_csv(path2Data)['age']
for i in np.arange(2,len(columns)):
    synsemVsAge[columns[i]]=np.repeat(np.nan,len(subjects))
for files in synsemFiles1[1:]:
    subPos=np.argwhere(subjects=='sub_'+files[_Pos:_Pos+8])
    if subPos.shape==(1,1) and files[-10:-4]!='failed':
        scoreError=np.array(pd.read_csv(path2txt+path2synsem1+files, sep='\t'))[0]
        scorePvalid=np.array(pd.read_csv(path2txt+path2synsem2+files[:-11]+'.txt', sep='\t'))[0]
        synsemVsAge.loc[subPos[0][0],columns[2]]=np.array(scorePvalid[1])
        synsemVsAge.loc[subPos[0][0],columns[3]]=np.mean(scoreError[21:31])
        

df_melt=pd.melt(synsemVsAge,id_vars=['Age'],value_vars=['pValid','error'])

color='mako'
sns.displot(data=df_melt,x='value', hue='variable',kind='kde', 
            fill=True,palette=color)
sns.lmplot(x ='Age', y ='value', data = df_melt, hue ='variable', 
            scatter=True,scatter_kws={'alpha':0.3}, palette =color)

scoreDf['Synsem']=synsemVsAge['error']

#%% VSTM: Visual short-term memory
# View (1–4) coloured discs brieﬂy presented on a computer screen,
# then after a delay, attempt to remember the colour of the disc that
# was at a cued location, with response indicated by selecting the
# colour on a colour wheel (touchscreen input).
# Para entender los scores-referirse a https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6084281/pdf/pag_33_5_841.pdf
path2vstm='VSTMcolour/release001/data/'
VSTMFiles=np.sort(os.listdir(path2txt+path2vstm))[1:]
_Pos=VSTMFiles[0].find('_')+1 #find where the subject ID starts 
columns=['ID','Age','K','Precision','K1','K2','K3','K4','P1','P2','P3','P4']
vstmVsAge=pd.DataFrame(index=range(len(subjects)),columns=columns)
vstmVsAge['ID']=pd.read_csv(path2Data)['CCID']
vstmVsAge['Age']=pd.read_csv(path2Data)['age']
for i in np.arange(2,len(columns)):
    vstmVsAge[columns[i]]=np.repeat(np.nan,len(subjects))
for files in VSTMFiles:
    subPos=np.argwhere(subjects=='sub_'+files[_Pos:_Pos+8])
    if subPos.shape==(1,1) and files[-10:-4]!='failed':
        score=np.array(pd.read_csv(path2txt+path2vstm+files, sep='\t').iloc[0,:-1]).astype(float)
        if str(type(score))=="<class 'numpy.ndarray'>":
            if not(np.isnan(np.array(score)).any()):
                # vstmVsAge.loc[subPos[0][0],'K']=score[[1,6,12,18]]/[1,2,3,4] #devide by the test-size to get a probability from the von-mises dist
                # vstmVsAge.loc[subPos[0][0],'Precision']=score[0,5,11,17]
                vstmVsAge.loc[subPos[0][0],'K1']=score[1]/1
                vstmVsAge.loc[subPos[0][0],'K2']=score[6]/2
                vstmVsAge.loc[subPos[0][0],'K3']=score[12]/3
                vstmVsAge.loc[subPos[0][0],'K4']=score[18]/4
                vstmVsAge.loc[subPos[0][0],'P1']=score[0]
                vstmVsAge.loc[subPos[0][0],'P2']=score[5]
                vstmVsAge.loc[subPos[0][0],'P3']=score[11]
                vstmVsAge.loc[subPos[0][0],'P4']=score[17]

df_melt=pd.melt(vstmVsAge,id_vars=['Age'],value_vars=['K1','K2','K3','K4'])
color='CMRmap'
sns.displot(data=df_melt,x='value', hue='variable',kind='kde', 
            fill=True,palette=color)
sns.lmplot(x ='Age', y ='value', data = df_melt, hue ='variable', 
           scatter=True,scatter_kws={'alpha':0.3}, palette =color)

scoreDf['VSTM']=vstmVsAge['K4']
#%% Get if we have PSD, Anat and FC

path2fc='/home/isaac/Documents/Doctorado_CIC/Internship/Sylvain/New_thesis/camcan_AEC_ortho_Matrix'
fcDir=np.sort(os.listdir(path2fc))
_Pos=fcDir[0].find('-')+1 #find where the subject ID starts 

for files in fcDir: 
    subPos=np.argwhere(subjects=='sub_'+files[_Pos:_Pos+8])
    scoreDf.loc[subPos[0][0],'FC']=1
    
    
#%%
import pickle
with open('/home/isaac/Documents/Doctorado_CIC/Internship/Sylvain/New_thesis/Python_Fun/scoreDf.pickle', 'wb') as f:
    pickle.dump(scoreDf, f)