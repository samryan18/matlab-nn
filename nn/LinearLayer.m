%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Samuel Ryan
% Date: February 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef LinearLayer < handle % handle makes these objects pass by reference
    properties (SetAccess = public)
        parameters
        gradients
        inputs
    end
    
    methods
        function L = LinearLayer(input_size, output_size, ...
                                w_init, ... % OPTIONAL PARAM
                                b_init... % OPTIONAL PARAM
                                )
            L.parameters = containers.Map;
            L.gradients = containers.Map;

            % check for weight and biases initializations
            if ~exist('w_init','var')
                w_init = rand(input_size, output_size)-1;
            elseif ~isequal(size(w_init),[input_size, output_size])
                error(strcat('Incorrect dimension on weight tensor ',...
                             'initialization for linear layer. Was shape: ', ...
                              mat2str(size(w_init)), ' Needed: ', ...
                              mat2str([input_size, output_size]),'.'))

            end

            if ~exist('b_init','var')
                b_init = rand(1, output_size)-1;
            elseif ~isequal(size(b_init),[1, output_size])
                error(strcat('Incorrect dimension on bias tensor ', ...
                             'initialization ', ...
                             'for linear layer. Was shape: ', ...
                             mat2str(size(b_init)), ' Needed: ', ...
                             mat2str([1, output_size]),'.'))
            end

            L.parameters('w') = w_init;
            L.parameters('b') = b_init;
        end

        function output = forward(layer, inputs)
            layer.inputs = inputs;
            output =  inputs * layer.parameters('w') + layer.parameters('b');
        end

        function grad = backward(layer, gradient)
            layer.gradients('b') = sum(gradient).';
            layer.gradients('w') = gradient.' * layer.inputs;
            grad = (layer.parameters('w')) * gradient.';
        end
    end
end