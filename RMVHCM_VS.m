clear,clc;
%% IS
% Data = load('IS.txt');  Y = Data(:,end); Data(:,end) = [];
% Data = mapminmax(Data',0,1); Data = Data';
% view = 2; features = [9 10];
% left = 1;right=0; X=cell(1,view);
% for i=1:view
%     X{i} = Data(:, left:features(i)+right);
%     left = features(i)+right+1;
%     right = features(i)+right;
% end
% data=X;
% truelabel{1}=Y;
% H=view;
% C=max(Y);

%% 100Leaves
% load 100Leaves.mat
% data=X;truelabel{1}=Y;
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=data{h}';
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end


%% forest tpye
Data = load('foresttype.txt');  Y = Data(:,end);
Data = mapminmax(Data',0,1); Data = Data';Data(:,end) = Y;
view = 2; features = [9 18];

left = 1;right=0; X=cell(1,view);
for i=1:view
    X{i} = Data(:, left:features(i)+right);
    left = features(i)+right+1;
    right = features(i)+right;
end
data=X;
truelabel{1}=Y;
C=max(truelabel{1});
H=view;

%% MNIST-10k
% load MNIST-10k.mat
% data=X;
% truelabel{1}=Y;
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end

%% motion
% load motion.mat
% data={X_person1,X_person2};
% truelabel{1}=Y_person1';
% C=max(truelabel{1});
% [~,H]=size(data);
% features=[48,48];
% for h=1:H
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end



%% initialization
[n,~]=size(data{1});
target=cell(16);

times=10;
AC1=zeros(times,1);
nmi=zeros(times,1);
P=zeros(times,1);
R1=zeros(times,1);
F=zeros(times,1);
RI=zeros(times,1);
FM=zeros(times,1);
J=zeros(times,1);
metrics=[];
stdmetrics=[];

for alpha_i =1:16
    if(alpha_i==1)
        alpha=1.1;
    else
        alpha = (alpha_i-1)*2;
    end

    for  time=1:times
        lambda=zeros(1,H);
        U=zeros(n,C);
        W=zeros(1,H);
        center = cell(1,H);
        target{alpha_i}=0;
        for h=1:H
            lambda(h)=getBeta(data{h});
            col=size(data{h},2);
            center{h}=zeros(C,col);
            W(h)=1/H;
        end
        %% Initialize the cluster center
        for h=1:H
            center{h}=get_random_center(data{h},C);
        end
        %% Start Iteration
        loop=1;
        while true
            loop=loop+1;
            distance = zeros(n,C,H);
            new_center = cell(1,H);
            %% Calculate the distance
            for h =1:H
                for i = 1 : n
                    for j = 1 : C
                        distance(i,j,h)=alternative_metric(data{h}(i,:),center{h}(j,:),lambda(h));
                    end
                end
            end
            %% Updating  the membership
            for i=1:n
                dis=zeros(1,C);
                for j=1:C
                    U(i,j)=0;
                    for h=1:H
                        dis(j)=dis(j)+distance(i,j,h)*(W(h)^alpha);
                    end
                end
                [~,p]=min(dis);
                U(i,p)=1;
            end
            %% Updating the clusters
            for j=1:C
                for h = 1 : H
                    col=size(data{h}(1,:),2);
                    fz=zeros(1,col);
                    fm=0;
                    for i =1 : n
                        fz=fz+U(i,j)*exp(-lambda(h)*norm(data{h}(i,:)-center{h}(j,:))^2)*data{h}(i,:);
                        fm=fm+U(i,j)*exp(-lambda(h)*norm(data{h}(i,:)-center{h}(j,:))^2);
                    end
                    if(fm==0)
                        new_center{h}(j,:)=center{h}(j,:);
                    else
                        new_center{h}(j,:)=fz/fm;
                    end
                end
            end
            center = new_center;
            %% Updating the view weights
            for h=1:H
                W(h)=0;
                for k =1:H
                    fz=sum(sum(U(:,:).*distance(:,:,h)));
                    fm=sum(sum(U(:,:).*distance(:,:,k)));
                    W(h)=W(h)+(fz/fm)^(1/(alpha-1));
                end
                W(h)=1/W(h);
            end
            %% Calculate the objective function
            new_target=0;
            for h =1 :H
                new_target=new_target+sum(sum(U.*distance(:,:,h)*W(h)^alpha));
            end
            if(abs(new_target-target{alpha_i}(loop-1))<1e-5&&loop>12)
                break;
            end
            target{alpha_i}(loop)=new_target;
        end
        %% Tag matching and calculate evaluation metrics
        label=zeros(n,1);
        for i=1:n
            [~,p]=max(U(i,:));
            label(i)=p;
        end
        result_label=label_map(label,truelabel{1});
        Y=truelabel{1};
        AC1(time) = length(find(Y == result_label))/length(Y);
    end

    meanAC=mean(AC1);
    stdAC=std(AC1);
end



