function [beta] = getBeta(data)
    [n,m]=size(data);
    allcenter=zeros(1,m);
    for i= 1:n
        allcenter=allcenter+data(i,:);
    end
    allcenter=allcenter/n;
    distence=0;
    for i= 1:n
        distence=distence+norm(data(i,:)-allcenter)^2;
    end
    beta=n/distence;
end

