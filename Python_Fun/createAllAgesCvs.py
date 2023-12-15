import pandas as pd

current_path = os.getcwd()
parentPath = os.path.abspath(os.path.join(current_path, '../../'))
path2AllAges= parentPath +'/NewThesis_db_DK/camcan_demographics/AllAges.csv'
AllTheAges=pd.read_csv(path2AllAges)
AllTheAges = [AllTheAges.loc[i].iloc[0].split(',') for i in range(len(AllTheAges))]
for s,sub in enumerate(AllTheAges):
    sub[0] = 'sub_'+sub[0]
    AllTheAges[s] = sub
AllTheAges_df = pd.DataFrame(AllTheAges, columns=['CCID','Age','Sex','Hand','Coil','MT_TR'])
AllTheAges_df.drop(['Hand','Coil','MT_TR'], axis='columns', inplace=True)
AllTheAges_df.to_csv(parentPath +'/NewThesis_db_DK/camcan_demographics/AllTheAges.csv', index=None)