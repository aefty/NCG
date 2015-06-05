
N =10;
x0 = ones(1,N)*2.3;
graph = zeros(N,N);
base = grad(x0);
eps = 1;
v=0;

for i = 1:N
    x0(i) = x0(i) +eps;
    graph(:,i) = (grad(x0)-base);
   % (grad(x0)-base)
   % pause
   
    x0(i) = x0(i) -eps;
end

graph;
graph = graph.^2;

graph(graph>0.1)=1;
graph(graph~=1)=0

sum(sum(graph))



