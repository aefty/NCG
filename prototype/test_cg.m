clear; clc; close all;

% MulitD Rosenbrock initial Guess
x0=[7;7;7;7];
%x0=[-20;2];
[x,itr,history,t] = cg(x0);

itr
x
history


%General Plots
if(1) % Plot error
    f =history - ones(size(history));
	normValue = sum(f.^2,2).^(.5);

	hFig = figure;
	semilogx(1:size(normValue,1),normValue,'-sr');

	title('Rosenbrock - Convergence of NLCG');
	legend('NLCG Method');
	xlabel('Iteration') % x-axis label
	ylabel('Norm Convergence') % y-axis label
	grid on;
    set(hFig, 'Position', [100 100 350 400]);
end

if(0)% Plot Time
	figure;
	plot(t');
	title('NLCG Block Timing');
	legend('Line Search',...
    'Direction Calculation',...
    'Vector Transfer & Tolerance');
	xlabel('Iteration') % x-axis label
	ylabel('Time') % y-axis label
	grid on;
end

if(0)%% Plot Solver Path
	plotRange=[30,-30];

	phi = @(x1,x2) 100*(x2-x1.^2).^2+(1-x1).^2;
	X1 = linspace(-max(plotRange),max(plotRange),50);
	X2 = linspace(-max(plotRange),max(plotRange),50);
	[X1 X2] = meshgrid(X1,X2);
	R = phi(X1,X2);

	Xa=history(:,1);
	Ya=history(:,2);
	Za=phi(Xa,Ya);

	figure('Units','normalized','Position',[0 0 1 1])
	surf(X1,X2,R,'EdgeColor',[.1 .1 .1])
	hold on
	plot3(Xa,Ya,Za, '--rd',...
	    'LineWidth',2,...
	    'MarkerSize',10);
	hold on

	legend('Function','CG');
	title('100*(x(2)-x(1)^2)^2+(1-x(1))^2 vs Solvers C_1=1e-4')
end