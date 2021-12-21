tic;
Net_read = readtable('Network.csv');
Network=table2array(Net_read);
Dem_read = readtable('Demand.csv');
Demand=table2array(Dem_read);
fft=[]';
fft=Network(:,5);
cap=[]';
cap=Network(:,3);
K= digraph (Network(:,1), Network(:,2), fft);

Dem_matrix =zeros (24,24);
for i = 1:24
    for j = 1 : 24
        for n= 1 : 576
            if Demand (n,1)== i && Demand (n,2)== j
                Dem_matrix (i,j) = Demand(n,3);
            end
        end
    end
end

volx=zeros(76,1);
for i = 1:24
    for j= 1:24
        [p]= shortestpath (K,i,j);
         u=length(p);
            for n= 1 :u-1
                for l=1:76
                    if Network(l,1)==p(n) && Network(l,2)==p(n+1)
                        volx(l)=volx(l)+Dem_matrix(i,j);
                    end
                end
            end
    end
end

[zold, voly]= frank(Dem_matrix, Network, volx,cap,fft);
[znew,volx]= gold(volx,voly,cap,fft);
iteration=2;
conv=abs(zold-znew)/zold;
while conv > 0.000005
   [zold, voly]= frank(Dem_matrix, Network, volx,cap,fft);
   [znew,volx]= gold(volx,voly,cap,fft);
    conv=abs(zold-znew)/zold;
    iteration=iteration+1;
end

for i = 1:76
    tt(i)=fft(i)*(1+0.15*(volx(i)/cap(i))^4);
end

display (iteration)
Solution=table;
Solution.Initial_Node= Network(:,1);
Solution.End_Node= Network(:,2);
Solution.Volume= volx;
Solution.Cost= tt';
writetable(Solution,'Solution.xlsx')
toc

 
      