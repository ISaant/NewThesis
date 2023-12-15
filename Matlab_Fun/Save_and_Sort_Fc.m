path = '/media/isaac/Elements/camcan_AEC_ortho/';
path2save_AP='/home/isaac/Documents/Doctorado_CIC/NewThesis_db_DK/camcan_AEC_ortho_AnteroPosterior/';
path2save_YEO='/home/isaac/Documents/Doctorado_CIC/NewThesis_db_DK/camcan_AEC_ortho_YEO/';
names=readtable('/home/isaac/Documents/Doctorado_CIC/NewThesis_db_DK/camcan_AEC_ortho_YEO/Regions_and_Networks.csv');
Idx4SortingLR=load("Index2Sort_Anterioposterior.mat").Index;% Los indices son anteroposterior, pero hay que separar izquiera y derecha
Idx4SortingAP=[Idx4SortingLR(1:2:end),Idx4SortingLR(2:2:end)];
Idx4SortingYEO=table2array(load("Index2Sort_YEO.mat").sorted(:,'oldindex'));
namesLR=table2array(names(Idx4SortingAP,'name'));
namesYEO=table2array(load("Index2Sort_YEO.mat").sorted(:,'name'));
Dir=dir(path);
Dir= Dir(3:end);
[Fc_mean_AP,Freqs]=sort_matrix(path,Dir,Idx4SortingAP,path2save_AP);
plotandsave(Fc_mean_AP,namesLR,'AP_LR',Freqs,[34],[34])
[Fc_mean_YEO,Freqs]=sort_matrix(path,Dir,Idx4SortingYEO,path2save_YEO);
plotandsave(Fc_mean_YEO,namesYEO,'YEO',Freqs,[18,20,23,35,46,57,68],[18,20,23,35,46,57,68])
%%
function [Fc_mean,Freqs]=sort_matrix(path,Dir,Idx,path2save)
Fc_mean=zeros(68,68,6);
    for i=1:length(Dir)
    
        clear TF
        file=Dir(i).name;
        load(strcat(path,file));
        TF_Expand=process_compress_sym('Expand',TF,68); %la fucion es de brainstorm
        TF_Expand_Matrix=reshape(TF_Expand,[68,68,6]);
        TF_Expand_Matrix_Sorted=TF_Expand_Matrix(Idx,Idx,:);
        for j = 1:68
            TF_Expand_Matrix_Sorted(j,j,:)=nan;
        end
        Fc_mean=Fc_mean+TF_Expand_Matrix_Sorted;
        Rows=RowNames(Idx);
        filename=strcat(path2save,file(1:12),'.mat');
        if not(isfile(filename))
            save(filename,'TF_Expand_Matrix_Sorted','Rows','Freqs')
        else
            strcat('ya existe ', string(i))
        end
        
    
    end
    Fc_mean=Fc_mean/length(Dir);
end

function plotandsave(Fc_mean,names,filename,Freqs,col,row)
    path2saveMean='/home/isaac/Documents/Doctorado_CIC/NewThesis/Matlab_Fun/Matlab_Figures/';
    for i = 1:6
        Fc_z(:,:,i)= (Fc_mean(:,:,i) - nanmean(Fc_mean(:,:,i),'all'))/nanstd(Fc_mean(:,:,i),0,'all');
    end 
%     for i = 1:68
%     
%         Fc_z(i,i,:)=nan;
%     
%     end
    
    
    for i=1:6
    
        f=figure();
        f.Position = [100 100 1500 1500];
        hm=heatmap(Fc_z(:,:,i),'XData',names,'YData',names);
        colormap('jet')
        title(Freqs(i,1))
        % Get underlying axis handle
        origState = warning('query', 'MATLAB:structOnObject');
        cleanup = onCleanup(@()warning(origState));
        warning('off','MATLAB:structOnObject')
        S = struct(hm); % Undocumented
        ax = S.Axes;    % Undocumented
        clear('cleanup')
        hm.GridVisible = 'off';
        xline(ax, [col+.5], 'k-','LineWidth',4); % see footnotes [1,2]
        yline(ax, [row+.5], 'k-','LineWidth',4); % see footnotes [1,2]
        pause(1)
        saveas(hm,[path2saveMean,filename,'_',cell2mat(Freqs(i,1)),'.png'])
        
    end
end