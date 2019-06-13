# ~ Deep Learning ~
Feed Forward MLP Library with Matlab

* see `nn/` for library code
* see `nn_regression_example.m` for an example of fitting a 10 degree polynomial
* see `nn_classification_example.m` for more practical example (data not included)

---

### Available Loss Functions
* Mean squared error: 
```
mean_squared_error = @(y_hat,y) (1/size(y_hat,1)) * sum((y_hat-y).^2);
mean_squared_error_gradient = @(y_hat,y) (1/size(y_hat,1))*2*(y_hat-y);
```

* Cross entropy loss: 
```
cross_entropy_loss = @(y_hat,y) (1/size(y_hat,1)) * ...
                                         sum(-(y .* log(y_hat)) ...
                                             - (1-y) .* log(1-y_hat));
cross_entropy_loss_gradient = @(y_hat,y) (1/size(y_hat,1)) * (y_hat-y);
```

Add custom loss functions in `nn/ErrorFunctions.m`.

---

### Available Activation Functions
* Tanh
```
tanh_activation = @(x) tanh(x);
tanh_activation_prime = @(x) 1-(tanh(x).^2);
```
* ReLU
```
relu_activation = @(x) x.*(x>0);
relu_activation_prime = @(x) 1.*(x>0);
```
* Sigmoid
```
sigmoid_activation = @(x) 1./(1.+exp(-x));
sigmoid_activation_prime = @(x) 1./(1.+exp(-x)).*(1-(1./(1.+exp(-x))));
```

Add custom activation functions in `nn/ActivationLayer.m`.

---

### Notes
* requires Matlab R2016b or later
* also Matlab sux

### How to write Matlab code in a text editor and run in a terminal shell

Add these lines to your `.bash_profile` or `.zshrc`:
```
export PATH=/Applications/Matlab_R20XXx.app/bin:$PATH
# or wherever matlab is on your computer
# if you don't know where matlab is, open the matlab gui and run 'matlabroot'

alias matlab="matlab -nodisplay -nosplash"
# now you can just run '$ matlab' to start a matlab session
```