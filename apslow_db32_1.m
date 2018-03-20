clc
clear all;
M=csvread('cyhtaps_10min.csv',1,1);%¶ÁÈëcsvÎÄ¼þ
wave_l = zeros(length(M),1);
% disp(M(1:10,1));

for  i=1:104
s=M(1+144*(i-1):144*i,1);
l_s=length(s);
% disp(l_s);
[cA1,cD1]=dwt(s,'db32');
% disp(cA1);
A1 = upcoef('a',cA1,'db32',1,l_s);
% disp(cA1)
D1 = upcoef('d',cD1,'db32',1,l_s);
wave_l(1+144*(i-1):144*i,1) = A1;
wave_2(1+144*(i-1):144*i,1) = D1;
end
subplot(3,1,1);plot(M(1:432,1));
subplot(3,1,2);plot(wave_l(1:432,1));
subplot(3,1,3);plot(wave_2(1:432,1));
% disp(wave_l)
csvwrite('apslow_db32_1.csv',wave_l);