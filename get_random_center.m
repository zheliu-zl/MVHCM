function [centers] = get_random_center(data,c)
[n,m] =size(data);
rd=randperm(n);
centers=zeros(c,m);
for i=1:c
    centers(i,:)=data(rd(i),:);
end
end
