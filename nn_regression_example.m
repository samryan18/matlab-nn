clear all; clc; addpath('./nn/'); addpath('./util/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
activation_type = 'tanh';
learning_rate = 0.002;
num_epochs = 1000;
print_frequency = 50;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% create a crazy polynomial to learn

% 1000 training examples, 100 features
X_train = rand(1000,100);

% 5000 test examples
X_test = rand(5000,100);

constants = rand(1, size(X_train,2));
polynomial = randi([1 10],1, size(X_train,2));


noise_train = (1.8-.2).*rand(size(X_train,1),1) + .2;
noise_test = (2-.0).*rand(size(X_test,1),1) + .0;

y_train = sum(X_train .* repmat(constants,size(X_train,1),1) ...
              + X_train .^ repmat(polynomial,size(X_train,1),1),2);
y_test = sum(X_test .* repmat(constants,size(X_test,1),1) ...
              + X_test .^ repmat(polynomial,size(X_test,1),1),2);

y_train = noise_train .* ...
                ActivationLayer.sigmoid_activation((y_train./mean(y_train))-1);
y_test = noise_test .* ...
                ActivationLayer.sigmoid_activation((y_test./mean(y_test))-1);

% create layers
layers = cell(5,1);
layers{1} = LinearLayer(size(X_train,2), 20);
layers{2} = ActivationLayer.make_tanh_activation_layer();
layers{3} = LinearLayer(20, 15);
layers{4} = ActivationLayer.make_tanh_activation_layer();
layers{5} = LinearLayer(15, 1);

% create MLP
nn = NN(layers, learning_rate);

loss = Loss(ErrorFunctions.mean_squared_error, ...
            ErrorFunctions.mean_squared_error_gradient);

[final_loss, training_losses, test_losses] = train(nn, X_train, ...
                                                   y_train, num_epochs, ...
                                                   loss, X_test, y_test, ...
                                                   print_frequency);

Util.plot_learning_curve(training_losses, test_losses, num_epochs, ...
                         strcat(activation_type, ': Regression on ', ...
                                ' degree 10 polynomial function with noise', ...
                                ' (5 layer net)'));
