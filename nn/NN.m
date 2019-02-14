classdef NN < handle % handle makes these objects pass by reference
    properties (SetAccess = private)
        layers
        learning_rate
    end
    
    methods
        function nn = NN(layers, learning_rate)
            nn.layers = layers;
            nn.learning_rate = learning_rate;
        end

        function output = forward(nn, inputs)
            output = inputs;
            for i = 1:size(nn.layers,1)
                output = nn.layers{i}.forward(output);
            end

            % squash output
            % can customize this in the future (e.g. softmax)
            output = ActivationLayer.sigmoid_activation(output);
        end

        function gradient = backward(nn, gradient)
            for i = size(nn.layers,1) :  -1 : 1
                % iterate backwards to backpropogate gradients
                gradient = nn.layers{i}.backward(gradient);
            end
        end

        function step(nn)
            % gradient descent
            for i = 1:size(nn.layers)
                layer = nn.layers{i};
                k = keys(layer.parameters);
                vals = values(layer.parameters);
                for i = 1:length(layer.parameters)
                    key = k{i};
                    param = vals{i};
                    gradient = layer.gradients(key);
                    layer.parameters(key) = param ...
                                            - (nn.learning_rate .* gradient.');
                end
            end
        end

    end
end