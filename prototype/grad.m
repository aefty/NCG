function g = grad(point)

	h = 1e-6;
	N  = max(size(point));

	for(i=1:N)
		point_l = point;
		point_r = point;

		point_l(i) = point_l(i) - h;
		point_r(i) = point_r(i) + h;

	    g(i,1) = (func(point_r) - func(point_l))/(2*h);
	end
end