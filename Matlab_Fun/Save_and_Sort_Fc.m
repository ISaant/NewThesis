path = '/media/isaac/Elements/camcan_AEC_ortho/';
path2save='/home/isaac/Documents/Doctorado_CIC/Internship/Sylvain/New_thesis/camcan_AEC_ortho_Matrix/';
Idx4Sorting=load("Index2Sort_Anterioposterior.mat").Index;
Idx4SortingLR=[Idx4Sorting(1:2:end),Idx4Sorting(2:2:end)];
Dir=dir(path);
Dir= Dir(3:end);
Fc_mean=zeros(68,68,6);
%%
for i=1:length(Dir)

    clear TF
    file=Dir(i).name;
    load(strcat(path,file));
    TF_Expand=process_compress_sym('Expand',TF,68);
    TF_Expand_Matrix=reshape(TF_Expand,[68,68,6]);
    TF_Expand_Matrix_Sorted=TF_Expand_Matrix(Idx4SortingLR,Idx4SortingLR,:);
    Fc_mean=Fc_mean+TF_Expand_Matrix_Sorted;
    Rows=RowNames(Idx4SortingLR);
    filename=strcat(path2save,file(1:12),'.mat');
%     if not(isfile(filename))
%         save(filename,'TF_Expand_Matrix_Sorted','Rows','Freqs')
%     else
%         strcat('ya existe ', string(i))
%     end
    

end

%%

Fc_mean=Fc_mean/length(Dir);
for i = 1:6
    Fc_z(:,:,i)= (Fc_mean(:,:,i) - mean(Fc_mean(:,:,i),'all'))/std(Fc_mean(:,:,i),0,'all');
end 
for i = 1:68

    Fc_z(i,i,:)=nan;

end


for i=1:6

    f=figure();
    f.Position = [100 100 1500 1500];
    heatmap(Fc_z(:,:,i));
    colormap('jet')
    title(Freqs(i,1))

end