import os
for file in FcFile:
    newfilename = file[:3] + '_' + file[4:-4] + '_fc' + file[-4:]
    src =  path2fc+'/'+ file
    dst =  path2fc+'/'+ newfilename

    print(file, newfilename)
    os.rename(src, dst)
#%%

