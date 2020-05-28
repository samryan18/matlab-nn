%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Samuel Ryan
% Date: February 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef Loss < handle % handle makes these objects pass by reference
    properties (SetAccess = public)
        func
        derivative
    end

    methods
        function self = Loss(loss_function, loss_function_grad)
            self.func = loss_function;
            self.derivative = loss_function_grad;
        end
    end
end
