
# RNN and LSTM Neural Networks

This reviews the backpropagation algorithm used in both RNN and LSTM networks. It duplicates Andrew Ng's course: Sequence Models at coursera.org closely.

## Notations [1]:
- Superscript <a href="https://www.codecogs.com/eqnedit.php?latex=[l]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[l]" title="[l]" /></a> denotes an object associated with the <a href="https://www.codecogs.com/eqnedit.php?latex=l{^{th}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l{^{th}}" title="l{^{th}}" /></a> layer. 
    - Example: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a^{[4]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a^{[4]}" title="a^{[4]}" /></a> is the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;4^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;4^{th}" title="4^{th}" /></a> layer activation. 

- Superscript <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;(i)" title="(i)" /></a> denotes an object associated with the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;i^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;i^{th}" title="i^{th}" /></a> example. 
    - Example: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x^{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{(i)}" title="x^{(i)}" /></a>  is the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;i^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;i^{th}" title="i^{th}" /></a> training example input.

- Superscript <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left&space;\langle&space;t&space;\right&space;\rangle" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\left&space;\langle&space;t&space;\right&space;\rangle" title="\left \langle t \right \rangle" /></a> denotes an object at the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;t^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;t^{th}" title="t^{th}" /></a> time-step. 
    - Example: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="x^{\left \langle t \right \rangle}" /></a> is the input x at the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;t^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;t^{th}" title="t^{th}" /></a> time-step. 
    
- Lowerscript *i* denotes the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;i^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;i^{th}" title="i^{th}" /></a> entry of a vector.
    - Example: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a_{i}^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a_{i}^{[l]}" title="a_{i}^{[l]}" /></a> denotes the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;i^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;i^{th}" title="i^{th}" /></a> entry of the activations in layer *l*.

## RNN Neural Network

### RNN Cell


![RNN CELL](https://github.com/niuers/LearningMachineLearning/blob/master/resources/rnn_cell.png)
Figure 1. RNN cell for a single time-step computation. Inputs are <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" title="a^{\left \langle t-1 \right \rangle}" /></a> (hidden state from previous time step) and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="x^{\left \langle t \right \rangle}" /></a> (input data at current time step *t*). The outputs are activation <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="a^{\left \langle t \right \rangle}" /></a>, which can also be used to predict <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="y^{\left \langle t \right \rangle}" /></a>.


### RNN Forward Propagation

![RNN Forward Diagram](https://github.com/niuers/LearningMachineLearning/blob/master/resources/RNN_diagrams.png)
Figure 2. RNN forward diagram. The input sequence <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbf{x}=\left&space;(&space;x^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;x^{\left&space;\langle&space;2&space;\right&space;\rangle},\cdots&space;,x^{\left&space;\langle&space;T_{x}&space;\right&space;\rangle}&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathbf{x}=\left&space;(&space;x^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;x^{\left&space;\langle&space;2&space;\right&space;\rangle},\cdots&space;,x^{\left&space;\langle&space;T_{x}&space;\right&space;\rangle}&space;\right&space;)" title="\mathbf{x}=\left ( x^{\left \langle 1 \right \rangle}, x^{\left \langle 2 \right \rangle},\cdots ,x^{\left \langle T_{x} \right \rangle} \right )" /></a> is carried over <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;T_{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;T_{x}" title="T_{x}" /></a> time steps. The network outputs <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbf{y}=\left&space;(&space;y^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;2&space;\right&space;\rangle},\cdots&space;,y^{\left&space;\langle&space;T_{x}&space;\right&space;\rangle}&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathbf{y}=\left&space;(&space;y^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;2&space;\right&space;\rangle},\cdots&space;,y^{\left&space;\langle&space;T_{x}&space;\right&space;\rangle}&space;\right&space;)" title="\mathbf{y}=\left ( y^{\left \langle 1 \right \rangle}, y^{\left \langle 2 \right \rangle},\cdots ,y^{\left \langle T_{x} \right \rangle} \right )" /></a>.

### RNN Backpropagation

## LSTM Neural Network

### LSTM Cell

### LSTM Forward

### LSTM BackPropagation




## References
1. Andrew Ng, [Sequence Models@Coursera.org](https://www.coursera.org/learn/nlp-sequence-models), 2018.
