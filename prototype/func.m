% Multi-Dim Rosenbrock
% Credit to -> http://www.gatsby.ucl.ac.uk/~edward/code/minimize/rosenbrock.m
function val = func(x)
	D = length(x);
	val = sum(100*(x(2:D)-x(1:D-1).^2).^2 + (1-x(1:D-1)).^2);

	%val = sum(x.^2);
end
