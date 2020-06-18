# Key Concepts

- **Parameters**:
  - weight: _w_
  - bias: _b_
  - activation: _a_
- **Hyperparameters**:
  - Learning Rate : _η_
    - Too small, takes a very long time to converge to global equilibrium
    - Too large, takes a very short time to converge to some equilibrium (not necessarily global)
    - Rule of thumb:
      - Start at 0.01 or 0.001, scale by a factor of 10 up and down, according to how quickly the loss decreases over each epoch.
- **Neurons**:
  - Sigmoid
  - Tanh
  - ReLU
  - Saturated neuron issue:
    - A neuron is considered saturated when the combination of its inputs and parameters produces extreme values of z = wx + b.
- **Layer types**:
  - Dense/FC
  - Softmax
- **Input layer**
- **Hidden layer**
- **Output layer**
- **Forward popagation**
- **Backward popagation**
- **Cost/Loss Function**

  - Quadratic (MSE)

    - <img src="https://latex.codecogs.com/gif.latex?\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^{2}"/>

      - <img src="https://latex.codecogs.com/gif.latex?y_i"/>: true label
      - <img src="https://latex.codecogs.com/gif.latex?\hat{y_i}"/>: network est. label

  - Cross Entropy Cost:
    - The larger the difference, the faster we can learn on that particular neuron.

- **Optimizers**:
  - Gradient descent:
    - can get computationaly expensive to train
  - Stochastic gradient descent:
    - can split data into small batches, rendering it better than Gradient Descent
    - `batch_size`: refers to the number of batches to split the data into
