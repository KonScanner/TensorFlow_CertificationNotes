# Brief Introduction

## Three Categories of AI

- **ANI: Artificial Narrow Intelligence**
  - e.g. Visual recog systems, self driving cars, real time machine translation systems and automated financial trading systems
- **AGI: Artificial General Intelligence**
  - e.g. A single algorithm that does all the above examples stated above
- **ASI: Artificial Super Intelligence**
  - More intelligent than a human and an AGI

## Traditional vs Deep learning Modeling Pipeline

- **TMLP** : 70+% Feature engineering + 30% Modeling
  - Spends time extracting features manually.
- **DLP** : 30+% Feature engineering + 70% Modeling
  - Spends time on having the model extract features automatically. (Representation Learning)

## Other Neural Networks

- **ANN: Artificial Neural Network**
  - Inspired\* by understanding of how the brain works.
  - A collection of artificial neurons so that they send and collect information between each other.
  - **DLN: Deep Learning Network**
    - Total of >5 layers of artificial neurons.
      - 1 input layer, 3+ hidden layers, 1 output layer
        - e.g. _Natural Language Processing_
- **CNN: Convolutional Neural Network**
  - e.g. Machine vision, Generative Adverserial Networks
- **DRL: Deep Reinforcement Learning**
  - When an artificial neural network including a deep neural network is involved in it.
    - Deep Learning algorithms handle all the data, while the Reinforcement learning Algorithm shines at selecting an appropriate action to take from a vast scope of possibilities.

## Factors that brough DL back in fashion:

- More data
- More compute power
- Theoretical advances in the field

### Play around with TF Playground:

http://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=8,8,4,2&seed=0.32263&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

## Biological Neurons resemblance to Perceptron:

- dendrites == input layer
- cell body == neuron
- action potential through axon == output
  - <img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}w_ix_i=wx+b>0"/> where _b_ (-ve) is the threshold then output 1 in the single layer perceptron
  - In this case the activation function is a _sigmoid_:
    - <img src="https://latex.codecogs.com/gif.latex?O(z)=\frac{1}{1+\exp{(-z)}}"/>
    - There are also:
      - <img src="https://latex.codecogs.com/gif.latex?O(z)=\tanh{z}"/>
      - <img src="https://latex.codecogs.com/gif.latex?O(z)=\max{(0,z)}"/>
    - Non linear nature of functions help deep learning Networks approximate any kind of problem.
      - RelU is the most efficient in compute (easier deriv.)
