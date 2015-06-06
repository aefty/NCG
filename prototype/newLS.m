clear all; close all; clc;

h = 100

range =10;
x = 0:range;

f = @(x)h*(1+x)/x

figure
scatter(f(x),ones(size(x)));

figure
plot(x,f(x));