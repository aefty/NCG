clear all; close all; clc;

figure
title('Exponential Estimate of Max()')
ezplot('max(1-x,0)',[-1,5])
hold on
ezplot('(x-1)/(exp((x-1)/.3)-1)',[-1,5])
ezplot('(x-1)/(exp((x-1)/.2)-1)',[-1,5])
ezplot('(x-1)/(exp((x-1)/.1)-1)',[-1,5])
grid on