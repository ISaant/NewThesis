import os
import numpy as np
current_path = os.getcwd()
parentPath = os.path.abspath(os.path.join(current_path, '../../'))
path2anat_s200 = parentPath+'/NewThesis_db_s200/stats-schaefer200x7_cvs/'
AnatFile_s200 = np.sort(os.listdir(path2anat_s200))

for file in AnatFile_s200:
    newfilename = file[:3] + '_' + file[4:]
    src =  path2anat_s200 + file
    dst =  path2anat_s200 + newfilename

    print(file, newfilename)
    os.rename(src, dst)
#%%

