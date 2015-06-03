%% CG Prototype - Non linear Conjugate gradient
%% 2015 - LMST Project
%% Aryan Eftekhari


function [x1,itr,history,t] = cg(x0)

	TOL = 1e-8;
	ITR = 10000;
	EPS = 1e-9;

	t=[];

	math = cuAlg();% This wont an object -  it will be a collection of inline functions


	%% Block 1 - Setup
	%% =================
    g0 = Grad(x0);
    p = -g0;

    %gg0 = g0'*g0;
    gg0 = math.vtv(g0,g0);
    itr =0;
    f=0;
	x1=x0; %Pointer Swap
	tol = TOL+1;

	history(1,:)=x0;%For testing only

	while (tol > TOL && ITR > itr)

		block2 = tic;
		%% Block 2 - Find Alpha
		%% ====================
		j=0; alpha_last=1; alpha=2;
        while (j<10 && abs(alpha-alpha_last)>=EPS)

        	%% Note : Calculate Hessian x p (Hp)
        	%% 2nd-term Taylor expansion (average -/+ expansion for better accuracy)

            %g00 = Grad(x1-EPS*p);
            vtemp = math.plus(1,x1,-TOL,p);
			g00 = Grad(vtemp);

            %g01 = Grad(x1+EPS*p);
            vtemp = math.plus(1,x1,TOL,p);
			g01 = Grad(vtemp);

            %Hp = (g01-g00)/(2*EPS);
            vtemp = math.plus(1,g01,-1,g00);
            Hp = math.vsm(1/(2*TOL),vtemp);

            alpha_last= alpha;

            %alpha = -g00'*p/(p'*Hp);
            stemp = math.vtv(g00,p);
            alpha = -1 * stemp / math.vtv(p,Hp);

            %x1=x1+alpha*p;
            x1 = math.plus(1,x1,alpha,p);

            j=j+1;
        end
        %% ===========================
        t(1,itr+1) = toc(block2)/j/9;

        brock3 = tic;
        % Block 3 - Next Direction
  		%% ========================
        g1=Grad(x1);

        %gg1 = g1'*g1;
     	gg1 = math.vtv(g1,g1);

        B = (gg1)/(gg0);
		%p = -g1+B*p;
     	p = math.plus(-1,g1,B,p);
		%% ===========================
     	t(2,itr+1) = toc(brock3)/3;

		brock4 = tic;
		%% Block 4 - Calculate tolerance finish iteration
		%% ===========================
		%tol = norm(x1-x0)
	    vtemp = math.plus(1,x1,-1,x0);
		tol = math.vtv(vtemp,vtemp)^(0.5); % In working code we drop sqrt and change TOL = sqrt(TOL)
		gg0=gg1;
		x0=x1;
		%% ===========================
		t(3,itr+1) = toc(brock4)/2;

		itr = itr+1;

		history(end+1,:) = x1;
	end
end


% Inline function
function g = Grad(point)

	h = 1e-8;
	N  = max(size(point));

	for(i=1:N)
		point_l = point;
		point_r = point;

		point_l(i) = point_l(i) - h;
		point_r(i) = point_r(i) + h;

	    g(i,1) = (func(point_r) - func(point_l))/(2*h);
	end
end
