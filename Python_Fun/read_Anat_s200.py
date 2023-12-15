import os
import numpy as np
from time import sleep
import pandas as pd
from freesurfer_stats import CorticalParcellationStats as readStats # freesurfer-stats 1.2.1 requires pandas<2,>=0.21, but you have pandas 2.1.4 which is incompatible.

import os
import numpy as np
current_path = os.getcwd()
parentPath = os.path.abspath(os.path.join(current_path, '../../'))
path2anat_s200 = parentPath+'/NewThesis_db_s200/stats-schaefer200x7/'
AnatFile_s200 = np.sort(os.listdir(path2anat_s200))
#%%

#WEY! SUPER IMPORTANTE! ELIMINSTE AL SUJETO CC221585 PORQUE NO TIENE CARACTERISTICAS ANATOMICAS DE UNO DE LOS HEMISFERIOS
for e,(left,right) in enumerate(zip(AnatFile_s200[::2],AnatFile_s200[1::2])):
    if  left[:12] != right[:12]:
        print('los archivos pares no son del mismo sujeto')
        break

    leftStats=np.array(readStats.read(path2anat_s200+left).structural_measurements.drop(index=0,columns='structure_name').reset_index(drop=True))
    rightStats=np.array(readStats.read(path2anat_s200+right).structural_measurements.drop(index=0,columns='structure_name').reset_index(drop=True))
    stats = pd.DataFrame(np.vstack((leftStats, rightStats)))
    path=os.path.join('/home/isaac/Documents/Doctorado_CIC/NewThesis_db_s200/stats-schaefer200x7_csv/'+left[:12]+'_anat.csv')
    stats.to_csv(path, index=False)


#%%

