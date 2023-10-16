toSort=readtable('/home/isaac/Documents/Doctorado_CIC/NewThesis_db/camcan_AEC_ortho_YEO/Regions_and_Networks.csv');
[sorted,index]=sortrows(toSort,"network");
sorted.oldindex=index;
save('/home/isaac/Documents/Doctorado_CIC/NewThesis/Matlab_Fun/Index2Sort_YEO.mat',"sorted")