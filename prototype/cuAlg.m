%% cuAlg Prototype -  Linear algebra
%% 2015 - LMST Project
%% Aryan Eftekhari

% Template of difference functions to be implementation cuda.
% Not all function will be required.
% Functions will be written as a collection of inline functions.

classdef  cuAlg
    properties (Access = 'public')
    end

    methods (Access = 'public')
        %Constructor
        function this = cuAlg()
        end

        %Vector^T Vector **
        function value = vtv(this,v1,v2)
            value= v1'*v2;
        end

        %Vector Addtions **
        function value = plus(this,c1,v1,c2,v2)
            value = c1*v1+c2*v2;
        end

        %Vector Scaler mult **
        function value = vsm(this,c,v)
            value = c*v;
        end
    end
end