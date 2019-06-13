clear all; clc; addpath('./nn/'); addpath('./util/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataset_name = 'Synthetic-Dataset'; % Data not included in repo
num_nodes = 50;
activation_type = 'sigmoid';
learning_rate = 0.1;
num_epochs = 8000;
print_frequency = 500;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load data
X_train = Util.load_data('X_train', dataset_name);
y_train = Util.load_data('y_train', dataset_name);
X_test = Util.load_data('X_test', dataset_name);
y_test = Util.load_data('y_test', dataset_name);

% convert to 0-1
y_train = (y_train+1)/2;
y_test = (y_test+1)/2;

% create layers
layers = cell(3,1);
layers{1} = LinearLayer(size(X_train,2), num_nodes);
if isequal(activation_type, 'sigmoid')
    layers{2} = ActivationLayer.make_sigmoid_activation_layer();
elseif isequal(activation_type, 'relu')
    layers{2} = ActivationLayer.make_relu_activation_layer();
elseif isequal(activation_type, 'tanh')
    layers{2} = ActivationLayer.make_tanh_activation_layer();
end
layers{3} = LinearLayer(num_nodes, 1);

% create MLP
nn = NN(layers, learning_rate);

loss = Loss(ErrorFunctions.cross_entropy_loss, ...
            ErrorFunctions.cross_entropy_loss_gradient);

[final_loss, training_losses, test_losses] = train(nn, X_train, ...
                                                   y_train, num_epochs, ...
                                                   loss, X_test, y_test, ...
                                                   print_frequency);

Util.plot_learning_curve(training_losses, test_losses, num_epochs, ...
                         strcat(activation_type, ': Learning Curve ', ...
                         ' (1 layer, ', num2str(num_nodes), ...
                         'hidden neurons) ', '(', dataset_name, ')'));

y_hat_tr = predict(nn, X_train);
y_hat_test = predict(nn, X_test);
train_error = ErrorFunctions.classification_error(y_train, y_hat_tr);
test_error = ErrorFunctions.classification_error(y_test, y_hat_test);

disp(strcat('Final train error: ', num2str(train_error)));
disp(strcat('Final validation error: ', num2str(test_error)));

function y_hat = predict(nn, inputs)
    output = nn.forward(inputs);
    y_hat = output>0.5;
end