classdef ActivationLayer < handle % handle makes these objects pass by reference
    properties (SetAccess = public)
        parameters
        gradients
        inputs
        func
        func_prime
    end
    
    methods
        function activationLayer = ActivationLayer(func, func_prime)
            activationLayer.parameters = containers.Map; % empty no parameters
            activationLayer.gradients = containers.Map; % empty no parameters
            activationLayer.func = func;
            activationLayer.func_prime = func_prime;

        end

        function output = forward(activationLayer, inputs)
            activationLayer.inputs = inputs;
            output = activationLayer.func(inputs);
        end

        function gradient = backward(activationLayer, gradient)
            gradient = activationLayer.func_prime(activationLayer.inputs) ...
                            .* gradient.';
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods for creating various types of activation function

    properties (Constant)
        % sigmoid activation functions
        sigmoid_activation = @(x) 1./(1.+exp(-x));
        sigmoid_activation_prime = @(x) 1./(1.+exp(-x)).*(1-(1./(1.+exp(-x))));

        % ReLU activation functions
        relu_activation = @(x) x.*(x>0);
        relu_activation_prime = @(x) 1.*(x>0);

        % tanh activation functions
        % not needed in this hwk but was curious
        tanh_activation = @(x) tanh(x);
        tanh_activation_prime = @(x) 1-(tanh(x).^2);
    end

    methods (Static)
        function layer = make_tanh_activation_layer()
            layer = ActivationLayer(ActivationLayer.tanh_activation, ...
                                    ActivationLayer.tanh_activation_prime);
        end

        function layer = make_sigmoid_activation_layer()
            layer = ActivationLayer(ActivationLayer.sigmoid_activation, ...
                                    ActivationLayer.sigmoid_activation_prime);
        end

        function layer = make_relu_activation_layer()
            layer = ActivationLayer(ActivationLayer.relu_activation, ...
                                    ActivationLayer.relu_activation_prime);
        end
    end
end