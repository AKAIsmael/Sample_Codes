function [znew,volx] = gold(volx,voly,cap,fft)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
r=0.5*(sqrt(5)-1);
 a=0;
 b=1;
 xl=(b-a)*(1-r)+a;
 xr=(b-a)*r+a;
 vl=zeros(76,1);
 vr=zeros(76,1);
 zvl=zeros(76,1);
 zvr=zeros(76,1);
 zn=zeros(76,1);
 

 
while (b-a) > 0.00005
     for i =1:76
         vl(i)=volx(i)+xl*(voly(i)-volx(i));
         vr(i)=volx(i)+xr*(voly(i)-volx(i));
         zvl(i)=fft(i)*vl(i)+0.03*(fft(i)/cap(i)^4)*(vl(i))^5;
         zvr(i)=fft(i)*vr(i)+0.03*(fft(i)/cap(i)^4)*(vr(i))^5;
         
         
     end
     Zl=sum(zvl);
     Zr=sum(zvr);
     if Zl <= Zr
         b=xr;
         xr=xl;
         xl=(b-a)*(1-r)+a;
     else
         a=xl;
         xl=xr;
         xr=(b-a)*r+a;
     end
  
end
alpha=(b+a)/2;
znew=(Zl+Zr)/2;



for i = 1:76
   volx(i)=(vl(i)+vr(i))/2;
   tt(i)=fft(i)*(1+0.15*(volx(i)/cap(i))^4);
 
end
end


