clear all; close all; clc;

h = 1

range =10;
x = 0:range;

f = @(x)h*x./(1+x)

figure
scatter(f(x),ones(size(x)));

figure
plot(x,f(x));