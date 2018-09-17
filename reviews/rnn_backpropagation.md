
# RNN and LSTM BackPropagation Algorithm

This reviews the backpropagation algorithm used in both RNN and LSTM networks. It duplicates Andrew Ng's course: Sequence Models at coursera.org closely.

## RNN BackPropagation Algorithm

### Notations [1]:
- Superscript <a href="https://www.codecogs.com/eqnedit.php?latex=[l]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[l]" title="[l]" /></a> denotes an object associated with the <a href="https://www.codecogs.com/eqnedit.php?latex=l{^{th}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l{^{th}}" title="l{^{th}}" /></a> layer. 
    - Example: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a^{[4]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a^{[4]}" title="a^{[4]}" /></a> is the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;4^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;4^{th}" title="4^{th}" /></a> layer activation. 

- Superscript <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;(i)" title="(i)" /></a> denotes an object associated with the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;i^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;i^{th}" title="i^{th}" /></a> example. 
    - Example: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x^{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{(i)}" title="x^{(i)}" /></a>  is the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;i^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;i^{th}" title="i^{th}" /></a> training example input.

- Superscript <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left&space;\langle&space;t&space;\right&space;\rangle" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\left&space;\langle&space;t&space;\right&space;\rangle" title="\left \langle t \right \rangle" /></a> denotes an object at the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;t^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;t^{th}" title="t^{th}" /></a> time-step. 
    - Example: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="x^{\left \langle t \right \rangle}" /></a> is the input x at the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;t^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;t^{th}" title="t^{th}" /></a> time-step. 
    
- Lowerscript *i* denotes the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;i^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;i^{th}" title="i^{th}" /></a> entry of a vector.
    - Example: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a_{i}^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a_{i}^{[l]}" title="a_{i}^{[l]}" /></a> denotes the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;i^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;i^{th}" title="i^{th}" /></a> entry of the activations in layer *l*.

### RNN Cell


![RNN CELL](https://github.com/niuers/LearningMachineLearning/blob/master/resources/rnn_cell.png)
Figure 1. RNN cell for a single time-step computation. Inputs are <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" title="a^{\left \langle t-1 \right \rangle}" /></a> (hidden state from previous time step) and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="x^{\left \langle t \right \rangle}" /></a> (input data at current time step *t*). The outputs are activation <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="a^{\left \langle t \right \rangle}" /></a>, which can also be used to predict <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="y^{\left \langle t \right \rangle}" /></a>.


### RNN Forward Propagation
### RNN Backpropagation

## LSTM BackPropagation Algorithm





## References
1. Andrew Ng, [Sequence Models@Coursera.org](https://www.coursera.org/learn/nlp-sequence-models), 2018.
