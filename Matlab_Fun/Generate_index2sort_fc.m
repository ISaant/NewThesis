Dk_Vertices=Cortex_1500.Atlas(4).Scouts;
Dk_Coordinates=Cortex_1500.Vertices;
ROIs={Dk_Vertices.Label};
Centroids = zeros(length(Dk_Vertices),3);

for i = 1:length(Centroids)

    Centroids(i,:)=mean(Dk_Coordinates(Dk_Vertices(i).Vertices,:));

end

plot3(Centroids(:,1),Centroids(:,2),Centroids(:,3),'o')

CentroidsL=Centroids(1:2:end,:);
CentroidsR=Centroids(2:2:end,:);
[CL,idx]=sortrows(CentroidsL,'descend');
CR=CentroidsR(idx,:);
idxLeft=(idx*2)-1; idxRight=idx*2;
Index=zeros(1,length(ROIs));
Index(1:2:end)=idxLeft;Index(2:2:end)=idxRight;
SortedROIs=ROIs(Index);
