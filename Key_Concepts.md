# Key Concepts

## **Parameters**:

- weight: _w_
- bias: _b_
- activation: _a_

## **Hyperparameters**:

- Learning Rate : _η_
  - Too small, takes a very long time to converge to global equilibrium
  - Too large, takes a very short time to converge to some equilibrium (not necessarily global)
  - Rule of thumb:
    - Start at 0.01 or 0.001, scale by a factor of 10 up and down, according to how quickly the loss decreases over each epoch.

## **Neurons**:

- Sigmoid
- Tanh
- ReLU
- _Saturated neuron_ issue:
  - A neuron is considered saturated when the combination of its inputs and parameters produces extreme values of:
    - `z` = `wx` + `b`.
- Rule of thumb with neurons per layer:
  - The more the neurons, the more your network is at risk of being more computationaly complex than needed.
  - The less the neurons and your networks accuracy will be held back imperceptively.

## **Weight initialization**:

- Although cross entropy attenuates the effects of neuron saturation, once paired with thoughtful weight initialization, will reduce the likelihood of _neuron saturation_ in the first place.

## **Layer types**:

- Dense/FC
- Softmax

## **Input layer**

- Once the data is preprocessed, it's fed to the neural network via this layer, for further processing.

## **Hidden layer**

- A layer between input and output layers. That's where neurons take in sets of weights and inputs to produce an output through their activation function.
- Rule of thumb for choosing # of hidden layers:
  - The more abstract your the ground truth value `y` you'd like to estimate with your network is, then it will be more helpful to add than remove parameters and vice versa.

## **Output layer**

- Last layer of neuron(s) that produces a given output prediction given the task.

## **Forward popagation**

- Input data is being fed _Forward_, then processed layer by layer given each layers activation function and passes it to the successive layer(s).

## **Backward popagation**

- It passes the gradient of the error function back from the last to the first layer, updating the weights

## **Cost/Loss Function**

- **Quadratic (MSE)**:

  - <img src="https://latex.codecogs.com/gif.latex?\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^{2}"/>

    - <img src="https://latex.codecogs.com/gif.latex?y_i"/>: true label
    - <img src="https://latex.codecogs.com/gif.latex?\hat{y_i}"/>: network est. label

- **Cross Entropy Cost**:
  - The larger the difference, the faster we can learn on that particular neuron.

## **Optimizers**:

- **Gradient descent**:
  - can get computationaly expensive to train
- **Stochastic gradient descent**:

  - Can split data into small batches, rendering it better than Gradient Descent
  - `batch_size`: refers to the number of batches to split the data into
  - By estimating the gradient multiple times on the mini batches, the noise is _smoothed out_ and we are able to avoid local minima.
    - If the `batch_size` is too large:
      - The estimate of the gradient of the Cost/Loss function is far more accurate. However the model is "at risk" of being trapped in a local minima.
    - If the `batch_size` is too small:
      - It could be excessively noise, (small amount of data is being used to calculate the gradient of the rest of the dataset), training will take longer and you're probably not exhausting all the compute you're provided with.
  - `batch_size` sweet spot:
    - Start with `batch_size` of 32:
      - If the mini batch is too large to fit in machine memory, try decreasing the `batch_size` in powers of 2. from 32 to 16.
      - If the model is training well, cost is going down consistently, but you're aware that you have _RAM_ memory on your machine, increase the `batch_size`.
      - To avoid getting trapped in local minima, avoid going > 128.

- **Unstable Gradients** issue:

  - _Vanishing gradients_:
    - As we back propagate from the last to the first layer, the gradient tends to "flatten" (vanishes). Because of this problem, if we naively kept adding layers to our network, eventually the hidden layers furthers from the output would not be able to learn to any extend, which _cripples_ the capacity of the network to further learn to approximate some relationship b/w `x` and `y`.
  - _Explosive gradients_:
    - In this case, the gradient between a given parameter relative to cost becomes increasingly steep as we move from the final hidden layer toward the first hidden layer. The gradient saturates the network by exceedingly increasing its `z` values (`z` = `wx` + `b`).
  - **Solution**: _Batch normalization_:

    - Takes the `a` activations (output from the preceeding layer), subtracts the mean and divides by the standard deviation (`σ`) to reshape the activations to a standard normal distribution. Thus if there are any extreme values in the preceeding layer, they won't cause _exploding_ or _vanishing_ gradients in the next layer.

      - Other positive things about batch normalization:
        - Allows layers to learn more independently from each other. Because large values in one layer, won't excessively influence the calculations in the next layer
        - Allows for selection of a higher _learning rate_. Because there are no extreme values in the normalized activation, thus overall the network can learn more quickly.
        - The layer outputs are normalized to the batch mean of the standard deviation. That adds a noise element, especially with smaller _batch sizes_, which in turn contributes to _regularization_\* (helps the network generalize, from its training data, to data that it hasn't encountered previously (like validation data)).
      - _Batch normalization_ adds **two extra learnable parameters** to any given layer it is applied to. In the final step of batch norm, the outputs are _linearly transformed_ by _multiplying_ by `γ` and adding `β`:
        - `γ` is analogous to `σ`
        - `β` is analogous to `μ`
        - This is the exact inverse of the operation that normalized the output values in the first place! However, the output values were originally normalized by the `batch` mean and `batch` standard deviation, whereas `γ` and `β` are learned by _SGD_.
        - The `batch` norm layer is initialized with `γ` = 1 and `β` = 0 and so, at the start of training,this linear transformation makes no changes; `batch norm` is allowed to normalize the outputs as intended. As the network learns, it may determine that _denormalizing_ any given layer's activations is optimal for reducing the cost.
        - In this way, if `batch` norm is **not** helpful, the network will learn to stop using it on a layer-by-layer basis. Because `γ` and `β` are continuous variables, the network can decide to what degree it would like to denormalize the outputs, depending on what works best to minimize the cost! 😎

## **Regularization** (avoiding overfitting):

- The situation, where _training cost_, continues to go **down**, while the _validation cost_ goes **up**, is called _overfitting_. Essentially learns the exact feature of the training data too closely and subsequently performs poorly on new, unseen data.
- **SOLUTIONS**:
  - **L1/L2 Regularization**:
    - _LASSO regression_ and _ridge regression_ are commonly used in ML problems, to penalize the model for including parameters by adding the parameters to the model's cost function.
    - The **larger** a given parameter's size, the more that parameter adds to the cost function. Because of this, parameters are not retrained by the model, unless they _appreciably contribute_ to the _reduction_ of the difference between the model's **estimated** `ŷ` (output) and the **true** `y` (output). i.e., irrelevant/external parameters are trimmed/cut away/off.
    - The distinction between _L1_ and _L2_ _regularization_ is that _L1_’s additions to cost correspond to the _absolute value of parameter sizes_, whereas _L2_’s additions correspond to the _square of the absolute values_ (L1/L2 norm vector distances). The net effect of this is that _L1_ _regularization_ tends to lead to the inclusion of a _smaller number_ of _larger-sized parameters_ in the model, while _L2_ regularization tends to lead to the inclusion of a _larger number_ of _smaller-sized parameters_.
  - **Dropout**:
    - Even though _L1_ and _L2_ _regularization_ works well, practicioners tend to vafor the use of a neural-network-specific _regularization_ technique. This technique is called _Dropout_, (Developed by Geoff Hinton and was used in AlexNet). In a nutshell, dropout simply _pretends_ that a randomly selected proportion of the neurons in each layer _don't exist_ during each round of training. For each round, we _remove_ a specified portion of hidden layers by random selection. Instead of reining in parameters sizes toward zero (as _batch normalization_ did), _Dropout_ doesn't (directly) constrain how large a given parameter value can become.
    - Nonetheless, _Dropout_ is an effective _regularization_ technique, because it prevents any single neuron from becoming excessively influential within the network. Making it more challenging for some very specific aspect of the training data set to create any overly specific forward propagation pathway through the network. That is, because on any given round of training, neurons along that pathway could be removed. In this way, the model doesn't become over-reliant on certain features of the data to generate a good prediction.
    - **Rules of thumb** for choosing which layers to apply _Dropout_ to and how much of it to apply:
      - If your network is overfitting to your training data (i.e., your validation cost increases while your training cost goes down), then _Dropout_ is warranted somewhere in the network.
      - Even if your network isn’t obviously overfitting to your training data, adding some _Dropout_ to the network may improve validation accuracy—especially in later epochs of training.
      - Applying _Dropout_ to _all of the hidden layers_ in your network may be **overkill**.
        - If your network has a fair bit of depth, it may be sufficient to apply _Dropout_ solely to later layers in the network (the earliest layers may be harmlessly identifying features). To test this out, you could begin by applying _Dropout_ only to the final hidden layer and observing whether this is sufficient for curtailing overfitting; if not, add _Dropout_ to the next deepest layer, test it, and so on.
      - If your network is _struggling_ to _reduce validation cost_ or to repeat low validation costs attained when less _Dropout_ was applied, then you’ve added too much _Dropout_-pare it back! As with other hyperparameters, there is a Goldilocks zone for _Dropout_, too.
      - When it comes to **how much** of _Dropout_ to apply to any given layer, each network architecture behaves uniquely, so some experimentation is required.
        - Dropping out 20% up to 50% of the hidden-layer neurons in ML visuion applications, tends to provide the highest validation accuracies. In NLP, where individual words and phrases can convey particular significance, 20% to 30% of the neurons in any given layer tends to perform better.
  - **Data augmentation**:
    - In addition to _regularizing_ your model’s parameters to _reduce_ overfitting, another approach is to _increase_ the size of your training dataset.
      - If it is possible to inexpensively collect additional high-quality training data for the particular modeling problem you’re working on, then you should do so!
      - If not, _data augmentation_ will help increase the size of the dataset, i.e. artificially expanding the training data set.
    - Techniques for _data augmentation_:
      - Skewing the image (**shifting the mean**).
      - Blurring the image (**smoothing**).
      - Shifting the image by a few pixels (**translation**).
      - Applying **random noise** to the image.
      - **Rotating** the image slightly.

## **Initialization steps and training**:

- Initialize every neuron and its weights will be granted some random initial values
- When first epoch of training begins, we shuffle and divide the training batches (data) into mini batches of `batch_size` each.
  - The shuffling step is what puts the "Stochastic" into the Stochastic gradient dsecent meaning.
- Forward propagate `x` through the network to estimate the `y_i` and `ŷ` or `a_i`.
- Then use a _cost/loss function_ to calculate `C`, by comparing `y_i` and `ŷ` or `a_i`.
- To minimize cost and therrefore improve the estimate of `y` given `x` the gradient descent part of the SGD is performed, weights and biases are adjusted in proportionality to how much they are contributing to the cost. Then they are fed by _Backward propagation_ to continue the next epoch.

* Early stoppibng for epoch hyper params:
  https://keras.io/api/callbacks/early_stopping/

## Other tips:

- When validating a neural network model that was trained using _dropout_\*, or when making real-world inferences with such a network, we must take an _extra step_ first. During validation or inference, we would like to leverage the power of the full network, that is, its total complement of neurons.
- The issue is that, during training, we only ever used a subset of the neurons to forward propagate `x` through the network and estimate `ŷ`.
- If we were to naïvely carry out this forward propagation with suddenly all of the neurons, our `ŷ` would emerge confused:
- There are now _too many parameters_, and the totals after all the mathematical operations would be larger than expected.
- To compensate for the additional neurons, we must correspondingly adjust our neuron parameters downward.

  - e.g. If we had, say, dropped out half of the neurons in a hidden layer during training, then we would need to multiply the layer’s parameters by 0.5 _prior_ to validation or inference.
  - e.g. If for a hidden layer in which we dropped out 33.3 percent of the neurons during training, we then must multiply the layer’s parameters by 0.667 prior to validation.

        -If the probability of a given neuron being retained during training is p, then we multiply that neuron’s parameters by p prior to carrying out model validation or inference.

- Thankfully, `Keras` handles this parameter-adjustment process for us automatically. When working in other deep learning libraries (e.g., low-level `TensorFlow`), however, you may **need** to be mindful and remember to carry out these adjustments yourself.
