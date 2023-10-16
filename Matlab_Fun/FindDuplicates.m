path='/home/isaac/Documents/Doctorado_CIC/NewThesis_db/camcan_AEC_ortho_Matrix';
Dir=dir(path);
Dir={Dir(3:end).name};

Sub46=load(string(strcat(path,'/',Dir(46))));
Sub47=load(string(strcat(path,'/',Dir(47))));
Sub48=load(string(strcat(path,'/',Dir(48))));
Sub49=load(string(strcat(path,'/',Dir(49))));


%%

Sub46_M=Sub46.TF_Expand_Matrix_Sorted;
Sub47_M=Sub47.TF_Expand_Matrix_Sorted;
Sub48_M=Sub48.TF_Expand_Matrix_Sorted;
Sub49_M=Sub49.TF_Expand_Matrix_Sorted;

for j=1:68

    Sub46_M(j,j,:)=nan;
    Sub47_M(j,j,:)=nan;
    Sub48_M(j,j,:)=nan;
    Sub49_M(j,j,:)=nan;

end



for i=1:6

    f=figure;
    f.Position(3:4)=[1500,600];
    subplot(1,3,1)
    heatmap(Sub48_M(:,:,i),Colormap=jet)
    subplot(1,3,2)
    heatmap(Sub49_M(:,:,i),Colormap=jet)
    subplot(1,3,3)
    heatmap(Sub48_M(:,:,i)-Sub49_M(:,:,i),Colormap=jet)
end