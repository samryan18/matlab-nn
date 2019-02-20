classdef ErrorFunctions
    % a class to store methods
    properties (Constant)
        mean_squared_error = @(y_hat,y) (1/size(y_hat,1)) * sum((y_hat-y).^2);
        mean_squared_error_gradient = @(y_hat,y) (1/size(y_hat,1))*2*(y_hat-y);
        
        cross_entropy_loss = @(y_hat,y) (1/size(y_hat,1)) * ...
                                         sum(-(y .* log(y_hat)) ...
                                             - (1-y) .* log(1-y_hat));
        cross_entropy_loss_gradient = @(y_hat,y) (1/size(y_hat,1)) * (y_hat-y);
        
        classification_error = @(y_pred,y_true) ...
                            1 - length(find(y_pred == y_true)) / length(y_true);
    end
end