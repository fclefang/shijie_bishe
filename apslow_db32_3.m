clc
clear all;
M=csvread('cyhtaps_10min.csv',1,1);%读入csv文件
wave_l = zeros(length(M),1);

for  i=1:104
s=M(1+144*(i-1):144*i,1);
l_s=length(s);
[C,L]=wavedec(s,3,'db32');
% disp(cA1);
A3 = wrcoef('a',C,L,'db32',3);%近似
D1 = wrcoef('d',C,L,'db32',1);%1层细节
D2 = wrcoef('d',C,L,'db32',2);
D3 = wrcoef('d',C,L,'db32',3);
wave_l(1+144*(i-1):144*i,1) = A3;
wave_2(1+144*(i-1):144*i,1) = D1;
wave_3(1+144*(i-1):144*i,1) = D2;
wave_4(1+144*(i-1):144*i,1) = D3;
end
subplot(5,1,1);plot(M(1:432,1));
subplot(5,1,2);plot(wave_l(1:432,1));
title('Approximation A3')
subplot(5,1,3);plot(wave_2(1:432,1));
title('Detail D1');
subplot(5,1,4);plot(wave_3(1:432,1));
subplot(5,1,5);plot(wave_4(1:432,1));
% disp(wave_l)
csvwrite('apslow_db32_3.csv',wave_l);
