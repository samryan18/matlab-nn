# ~ Deep Learning ~
Feed Forward MLP Library with Matlab

* see `nn/` for library code
* `nn_regression_example.m` has example of fitting a 10 degree polynomial
* see `main.m` for more practical example (data not included)

### Available Loss Functions
* Mean squared error
* Cross entropy loss

Add custom loss functions in `ErrorFunctions.m`.

### Available Activation Functions (see `ActivationLayer.m`)
* Tanh
* ReLU
* Sigmoid

Add custom activation functions in `ErrorFunctions.m`.

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