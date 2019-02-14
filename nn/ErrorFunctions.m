classdef ErrorFunctions
    % a class to store methods
    properties (Constant)
        mean_squared_error = @(y_hat,y) (1/size(y_hat,1)) * sum((y_hat-y).^2);
        mean_squared_error_gradient = @(y_hat,y) (1/size(y_hat,1))*2*(y_hat-y);
        
        cross_entropy_loss = @(y_hat,y) (1/size(y_hat,1)) * ...
                                         sum(-(y .* log(y_hat)) ...
                                             - (1-y) .* log(1-y_hat));
        cross_entropy_loss_gradient = @(y_hat,y) (1/size(y_hat,1)) * (y_hat-y);
    end

    methods (Static)
        function err = classification_error(y, y_pred)
            err = 1 - length(find(y_pred == y)) / length(y);
        end
    end
end