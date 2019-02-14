clear all; clc; addpath('./nn/'); addpath('./util/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
activation_type = 'tanh';
learning_rate = 0.002;
num_epochs = 600;
print_frequency = 50;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% create a crazy polynomial to learn

% 1000 training examples, 100 features
X_train = rand(1000,100);

% 5000 test examples
X_test = rand(5000,100);

constants = rand(1, size(X_train,2));
polynomial = randi([1 10],1, size(X_train,2));
y_train = sum(X_train .* repmat(constants,size(X_train,1),1) ...
              + X_train .^ repmat(polynomial,size(X_train,1),1),2);
y_test = sum(X_test .* repmat(constants,size(X_test,1),1) ...
              + X_test .^ repmat(polynomial,size(X_test,1),1),2);

y_train = ActivationLayer.sigmoid_activation((y_train./mean(y_train))-1);
y_test = ActivationLayer.sigmoid_activation((y_test./mean(y_test))-1);

% create layers
layers = cell(7,1);
layers{1} = LinearLayer(size(X_train,2), 20);
layers{2} = ActivationLayer.make_tanh_activation_layer();
layers{3} = LinearLayer(20, 30);
layers{4} = ActivationLayer.make_tanh_activation_layer();
layers{5} = LinearLayer(30, 20);
layers{6} = ActivationLayer.make_tanh_activation_layer();
layers{7} = LinearLayer(20, 1);

% create MLP
nn = NN(layers, learning_rate);

loss = Loss(ErrorFunctions.mean_squared_error, ...
            ErrorFunctions.mean_squared_error_gradient);

[final_loss, training_losses, test_losses] = train(nn, X_train, ...
                                                   y_train, num_epochs, ...
                                                   loss, X_test, y_test, ...
                                                   print_frequency);

Util.plot_learning_curve(training_losses, test_losses, num_epochs, ...
                         strcat(activation_type, ': EXAMPLE LEARNING CURVE '));
