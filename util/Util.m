classdef Util
    % a class to store random scrappy scripting methods

    properties (Constant)

    end

    methods (Static)

        function plot_learning_curve(training_losses, test_losses, ...
                                     num_epochs, my_title)
            % X = 100*(1:num_epochs/100);
            X = (1:num_epochs);
            fig = figure;

            plt = plot(X, training_losses, X, test_losses);

            plt(1).LineWidth = 2;
            plt(2).LineWidth = 2;
            xlabel('epoch');
            ylabel('loss');
            title(my_title);
            
            xtickangle(310);

            grid on;
            legend('Training Loss','Test Loss');
            saveas(fig, strcat('figures/', my_title, '.png'));
        end

        function [initializations] = load_weight_initializations(...
                                            activation_type, ...
                                            num_nodes, ...
                                            dataset_name)
            path = strcat('../../',dataset_name,'/');
            post = '.txt';
            % load weight initializations
            activation_path = strcat(activation_type, '/');

            initializations = containers.Map;

            pre = strcat(path, 'InitParams/', activation_path, ...
                         num2str(num_nodes));

            weight_initializations_path_1 = strcat(pre, '/w', num2str(1), post);
            bias_initializations_path_1 = strcat(pre, '/b', num2str(1), post);
            weight_initializations_path_2 = strcat(pre, '/w', num2str(2), post);
            bias_initializations_path_2 = strcat(pre, '/b', num2str(2), post);

            initializations('w1') = load(weight_initializations_path_1).';
            initializations('b1') = load(bias_initializations_path_1).';
            initializations('w2') = load(weight_initializations_path_2).';
            initializations('b2') = load(bias_initializations_path_2).';
        end
        
        function data = load_data(filename, dataset_name)
            path = strcat('../../',dataset_name,'/');
            post = ('.txt');
            full_path = strcat(path, filename, post);
            data = load(full_path);
        end
    end
end