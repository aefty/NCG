clear all; close all; clc;

h = 0.0001

range =10;
x = 0:range;

f = @(x)h*(1.5.^x-1)

scatter(f(x),ones(size(x)));