#%%
from importlib import reload
import os
from time import sleep
os.chdir('/home/isaac/Documents/Doctorado_CIC/NewThesis/Python_Fun')
# os.chdir('/export03/data/Santiago/NewThesis/Python_Fun')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from torch.utils.data import random_split
from FunClassifiers4newThesis_pytorch import *
from Fun4newThesis import *
from PSD_Features import PSD_Feat
from PSD_Features_s200 import PSD_Feat_s200
from Anat_Features import Anat_Feat
from Anat_Features_s200 import  Anat_Feat_s200
from read_Fc import read_Fc
from read_Sc import read_Sc
import Connectivity_Features
import node2vec_embedding
device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

# plt.ioff()

def Generate():
    #%%  Directories
    current_path = os.getcwd()
    parentPath = os.path.abspath(os.path.join(current_path, '../../'))
    path2psd = parentPath+'/NewThesis_db_DK/camcan_PSDs/'
    path2psd_s200 = parentPath+'/NewThesis_db_s200/camcan_PSDs'
    path2anat = parentPath+'/NewThesis_db_DK/camcan_Anat/'
    path2anat_s200 = parentPath+'/NewThesis_db_s200/stats-schaefer200x7_csv/'
    path2fc = parentPath+'/NewThesis_db_DK/camcan_AEC_ortho_AnteroPosterior'
    path2sc = parentPath+'/NewThesis_db_s200/msmtconnectome'
    path2demo = parentPath+'/NewThesis_db_DK/camcan_demographics/'
    AnatFile = np.sort(os.listdir(path2anat))
    AnatFile_s200 = np.sort(os.listdir(path2anat_s200))
    FcFile = np.sort(os.listdir(path2fc))
    ScFile = np.sort(os.listdir(path2sc))
    mainDir_psd = np.sort(os.listdir(path2psd))
    emptyRoomDir = np.sort(os.listdir(path2psd+mainDir_psd[0]+'/'))
    restStateDir = np.sort(os.listdir(path2psd+mainDir_psd[1]+'/'))
    PSDFile_s200 = np.sort(os.listdir(path2psd_s200))
    demoFile = np.sort(os.listdir(path2demo))

    #%% Find nan values in the score dataframe
    # import Fun4newThesis
    # reload(Fun4newThesis)

    print('Loading the Demographics and score table...')
    with open(current_path+'/scoreDf_spanish.pickle', 'rb') as f:
        scoreDf = pickle.load(f)

    #lets just keep age for now:
    # scoreDf.drop(columns=['Acer','BentonFaces','Cattell','EmotionRecog','Hotel','Ppp','Synsem','VSTM'],inplace=True)
    scoreDf.drop(columns=['BentonFaces','ReconocimientoEmociones', 'ForceMatch', 'Hotel', 'Ppp', 'Synsem',
           'VSTM'],inplace=True)
    row_idx=np.unique(np.where(np.isnan(scoreDf.iloc[:,3:-1].to_numpy()))[0])#rows where there is nan
    scoreDf_noNan=scoreDf.drop(row_idx).reset_index(drop=True)
    scoreDf_noNan=scoreDf_noNan.drop(np.argwhere(scoreDf_noNan['ID']=='sub_CC721434')[0][0]).reset_index(drop=True)# drop beacuase there is missing connections at the the struct connectomics
    PltDistDemographics(scoreDf_noNan)
    age=scoreDf_noNan['Edad'].to_numpy()

    with open(current_path+'/scoreDf.pickle', 'rb') as f:
        scoreDf_old = pickle.load(f)
    subjects=scoreDf_noNan['ID']
    subjects_old=scoreDf_old['ID']
    row_idx=[np.argwhere(subjects_old == missing)[0][0] for missing in list(set(subjects_old).difference(set(subjects)))] #Esto solo funciona para las matrices que te paso jason

    sleep(1)
    plt.close('all')
    #%% Hyperparameters
    #PSD
    print('Generating PSD (DK)...')
    freqs=np.arange(0,150,.5)
    freqs2use=[0,90]
    columns= [i for i, x in enumerate((freqs>=freqs2use[0]) & (freqs<freqs2use[1])) if x]
    freqsCropped=freqs[columns]
    #columns is used to select the region of the PSD we are interested in



    #%% Read PSD
    SortingIndex_AP = scipy.io.loadmat('/home/isaac/Documents/Doctorado_CIC/NewThesis/Matlab_Fun/Index2Sort_Anterioposterior.mat')['Index'].flatten()-1
    Idx4SortingAP=np.array([SortingIndex_AP[0::2],SortingIndex_AP[1::2]]).flatten()
    psd2use, restStatePCA=PSD_Feat (path2psd,mainDir_psd,restStateDir,emptyRoomDir,columns, row_idx, Idx4SortingAP)

    sleep(1)
    plt.close('all')
    #%% Read PSD s200
    print('Generating PSD (s200)...')
    psd2use_s200, restStatePCA_s200 = PSD_Feat_s200(path2psd_s200, PSDFile_s200, subjects)
    # subjectsID = [sub[:-8] for sub in PSDFile_s200]
    # set1 = set(list(subjects))
    # set2 = set(subjectsID)
    # missingSubjects = list(set1 - set2)
    # missingSubjects_idx=[np.where(subjects == sub)[0][0] for sub in missingSubjects]

    sleep(1)
    plt.close('all')
    #%% Read Anat & run stadistics
    print('Generating Cortical Features (DK)...')
    anat2use, anatPCA= Anat_Feat(path2anat,AnatFile,row_idx,scoreDf_noNan,Idx4SortingAP)
    sleep(1)
    plt.close('all')

    #%% Read Anat_s200 & run stadistics
    # reload(Anat_Features_s200)
    # from Anat_Features_s200 import Anat_Feat_s200
    print('Generating Cortical Features (s200)...')
    anat2use_s200, anatPCA_s200= Anat_Feat_s200(path2anat_s200, AnatFile_s200, scoreDf_noNan, subjects)
    sleep(1)
    plt.close('all')
    #%% Read Fc
    # You have to normalize this values for each matrix = take the max, min among all and (x-min(x))/max(x)
    # Aquí las regiones ya estan acomodadas por lo que no necesitas reacomodar usando Idx4SortingAP
    print('Reading Functional Connectivity (DK)...')
    boolarray=[x[4:-4]==y[4:] for x,y in zip(FcFile,subjects) ]
    print('All the subjects are sorted equal between the datasets: '+str(any(boolarray)) )

    # CorrHist(FcFile,path2fc)


    # min_perThresh_test(FcFile, path2fc)


    # delta, theta, alpha, beta, gamma_low, gamma_high, ROIs = Fc_Feat(FcFile,path2fc,thresh_vec[2])
    # delta, theta, alpha, beta, gamma_low, gamma_high, ROIs = read_Fc(FcFile,path2fc,.25) #nt = no threshold
    connectomes_fc, ROIs = read_Fc(FcFile,path2fc, subjects,40) #nt = no threshold
    # connectomes_nt, ROIs = read_Fc(FcFile,path2fc,subjects, 1) #nt = no threshold

    delta = connectomes_fc['delta']
    theta = connectomes_fc['theta']
    alpha = connectomes_fc['alpha']
    beta = connectomes_fc['beta']
    gamma_low = connectomes_fc['gamma_low']
    gamma_high = connectomes_fc['gamma_high']

    #%%
    rowlen = 68*6
    DiagFc = np.zeros((len(subjects), rowlen, rowlen))

    for e,_ in enumerate(subjects):
        data = [delta[e],theta[e],alpha[e],beta[e],gamma_low[e],gamma_high[e]] # merge for iteration

        col = 0
        for d in data: # each data list (A/B/C)
            DiagFc[e,col:col+len(d),col:col+len(d)] = d
            col += len(d)  # shift colu
    sns.heatmap(DiagFc[0,:,:])
    DiagFc = DiagFc/np.max(DiagFc)
    # connectomes_mod = kill_deadNodes(connectomes_fc)
    # alpha_mod, alpha_idx = connectomes_mod['alpha']
    # delta_mod, detla_idx=kill_deadNodes(delta)
    # theta_mod, theta_idx=kill_deadNodes(theta)

    # beta_mod, beta_idx=kill_deadNodes(beta)
    # gamma_low_mod, gl_idx=kill_deadNodes(gamma_low)
    # gamma_high_mod, gh_idx=kill_deadNodes(gamma_high)


    # ToDo hacer una clase con los atributos: num_nodes, num_edges, average node degree,
    # ToDo Probar si añadiendo una tansformacion laplaciana, la clasificacion mejora

    #%% Read Sc
    # import read_Sc
    # reload(read_Sc)
    # from read_Sc import read_Sc
    print('Reading Structural Connectivity (s200)...')
    connectomes, Length = read_Sc(ScFile,path2sc,subjects)

    #%%
    import Connectivity_Features
    # reload(Connectivity_Features)
    print('Generating Functional Connectivity Features (s200)...')

    if os.path.isfile('Schaefer200_Sc_Features.pickle'):
        with open('Schaefer200_Sc_Features.pickle','rb') as f:
            features_s200 = pickle.load(f)
        if len(features_s200[0])<len(connectomes):
            print(f'Computing features from subject: {len(features_s200[0])}')
            Connectivity_Features.traditionalMetrics(connectomes, Length, start_at=len(features_s200[0]))
        else:
            print('Sctructural features founded. No need to compute')
            local_s200 = np.array(features_s200[0])
            glob_s200 = np.array(features_s200[1])


    else:
        print('Sctructural features not found. Computing...')
        import Connectivity_Features
        reload (Connectivity_Features)
        Connectivity_Features.traditionalMetrics(connectomes,Length, start_at=0)

    Sub, Feat, ROI = local_s200.shape
    nPCA = 6
    local_PCA = np.zeros((Sub, nPCA, ROI))
    for roi in range(ROI):
        pca_df, pca2use, prop_varianza_acum = myPCA(local_s200[:, :, roi], False, nPCA)
        plt.plot(prop_varianza_acum)
        local_PCA[:, :, roi] = np.array(pca2use)

    local_PCA = RestoreShape(local_PCA)

    plt.figure()
    pca_df, glob_PCA, prop_varianza_acum = myPCA(glob_s200, False, 6)
    plt.plot(prop_varianza_acum)
    glob_PCA = np.array(glob_PCA)


    #%% Generate labels
    print('Generating Scores...')
    scores = np.array(scoreDf_noNan['Edad']).reshape(-1,1)



    return psd2use, restStatePCA, anat2use, anatPCA, DiagFc, restStatePCA_s200, anatPCA_s200, local_s200, glob_s200, ROIs, scores