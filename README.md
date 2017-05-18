# NFL_Win_Predictor
Neural Network in Chainer predicting wins in NFL

### Performance
As of 05/17/17 the entire test set of 2016 has a correct prediction rate of about 64%.  Selecting a higher confidence threshold brings the rate up, to 70/80% for 160/90 games respectively.
Shown below is a 70% correct prediction rate.
![prediction graph](https://github.com/jaymefosa/NFL_Win_Predictor/blob/master/Correct%20vs%20Incorrect.png)

### Neural Network Architecture 
This is made of two 100 neuron fully connected hidden layers, with dropout, gradient decay, and ReLU activations.
Many configurations were tested and the results are somewhat arbitrary, and well performing.

### Feature Selection
As there are many features to choose from and a quick check of the correlation coefficient didn't reveal anything to remove, 
I opted for ease of dynamic loading.  Currently in use are: Average yards, Avg points scored, avg points allowed, avg turnovers, avg sum of defeated opponent's scores, home, and bye weeks.
Note* Eliminating all of the variables, except home, and using team_ids instead led to similar performance..
