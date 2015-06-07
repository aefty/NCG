clear all; close all; clc;

h = 1e-4

range =16;
x = 0:range;

f = @(x)h*(2.^x-1)

figure

scatter(f(x),ones(size(x)));

figure
title('1D Exponential Grid')
plot(x,f(x));