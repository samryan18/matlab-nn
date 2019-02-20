function [final_loss, training_losses, test_losses] = ...
                                train(...
                                    net, ...
                                    X, ...
                                    y, ...
                                    num_epochs, ...
                                    loss,...
                                    X_test, ... % only used for learning curve
                                    y_test, ... % only used for learning curve
                                    print_frequency ... % print training/test at this interval
                                )
    training_losses = [];
    test_losses = [];

    for epoch = 1:num_epochs
        y_hat = net.forward(X);
        epoch_loss = loss.func(y_hat, y);

        grad = loss.derivative(y_hat, y);
        net.backward(grad);
        net.step();  
        y_hat_test = net.forward(X_test);
        test_loss = loss.func(y_hat_test, y_test);
        training_losses = [training_losses epoch_loss];
        test_losses = [test_losses test_loss]; 
        if mod(epoch, print_frequency) == 0
            disp (strcat('Epoch: ', num2str(epoch), ': ', ...
                         num2str(epoch_loss), ' (', ...
                         num2str(test_loss),')')); 
        end
    end
    final_loss = epoch_loss;
end