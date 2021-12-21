function [zol, zold, voly] = frank(Dem_matrix, Network, volx,cap,fft)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

Time=zeros(76,1);
for l = 1:76
    for i = 1:24
        for j= 1:24
            if Network(l,1)==i && Network(l,2)==j
                Time(l)=fft(l)*(1+0.15*(volx(l)/cap(l))^4);
            end
        end
    end
end



T= digraph (Network(:,1), Network(:,2), Time);
              

voly = zeros(76,1);
for i = 1:24
    for j= 1:24
        [p]= shortestpath (T,i,j);
         u=length(p);
            for n= 1 :u-1
                for l=1:76
                    if Network(l,1)==p(n) && Network(l,2)==p(n+1)
                        voly(l)=voly(l)+Dem_matrix(i,j);
                    end
                end
            end
    end
end

zol=zeros(76,1);
for i=1:76
    zol(i)=fft(i)*volx(i)+0.03*(fft(i)/cap(i)^4)*(volx(i))^5;
end
zold=sum(zol);
end