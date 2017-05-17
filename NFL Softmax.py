

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.plotly as py
import chainer.cuda
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import random
import pandas as pd



num_features = 14
train_size = 1802


SAVE = "yes"
SAVE = "no"
LOAD = "yes"
LOAD = "no"
ITERATIONS = 200#40 #make 1 for no training #120 seems nice


### this will be a softmax based win predictor
class winPredictor(Chain):
    def __init__(self, num_features):        
        super(winPredictor, self).__init__(
            h1 = L.Linear(num_features, 100),
            h2 = L.Linear(100,100),
            h3 = L.Linear(100,20),

            h4 = L.Linear(100,100), 
            h5 = L.Linear(100,100), 
            h6 = L.Linear(100,100), 
            h7 = L.Linear(100,100), 
            h8 = L.Linear(100,100),
            h9 = L.Linear(100,50),
            h10 = L.Linear(50,20),     
    
            out = L.Linear(100, 2),
            )
                
    def __call__(self, x, train=True):
        f1 = F.dropout(F.relu(self.h1(x)), ratio=.5, train=train)
        
        f2 = F.dropout(F.relu(self.h2(f1)), ratio=.5, train=train)
        
        #f3 = F.dropout(F.tanh(self.h3(f2))) 63% here
        
        """
        f4 = F.dropout(F.tanh(self.h4(f3)))
        f5 = F.dropout(F.tanh(self.h5(f4)))
        f6 = F.dropout(F.tanh(self.h6(f5)))
        f7 = F.dropout(F.tanh(self.h7(f6)))
        f8 = F.dropout(F.tanh(self.h8(f7)))
        f9 =  F.dropout(F.tanh(self.h9(f8)))
        f10 = F.dropout(F.tanh(self.h10(f9)))
        """
        y = (self.out(f2))
        return y

        
class Classifier(Chain):
     def __init__(self, predictor):
         super(Classifier, self).__init__(predictor=predictor)

     def __call__(self, x, t, train=True):

         loss = 0
         
         y = self.predictor(x)
         
         loss = F.softmax_cross_entropy(y, t)
         accuracy = F.accuracy(y, t)
         #print("loss", loss.data,
         #     "accuracy", accuracy.data)
         
         
         return loss, accuracy
        
         
         
         
network = winPredictor(num_features)        
model = Classifier(network)

optimizer = optimizers.Adam()
optimizer.setup(model)
model.to_gpu(0)
free_model = model.predictor
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

   
def randomSwapping(entire_array):
    for m in range(10):
        for k in range(len(entire_array)):
            choice = random.choice([True, False])
            if choice == True:
                entire_array[k][-1], entire_array[k][-2] = entire_array[k][-2], entire_array[k][-1]
            
                last_index = (len(entire_array[0]) - 2) / 2
        
                for i in range(last_index):
                    entire_array[k][i], entire_array[k][last_index + i] = entire_array[k][last_index + i], entire_array[k][i]

    return entire_array
    
def inverseSwapping(entire_array):
    for k in range(len(entire_array)):
            choice = True
            if choice == True:
                entire_array[k][-1], entire_array[k][-2] = entire_array[k][-2], entire_array[k][-1]
            
                last_index = (len(entire_array[0]) - 2) / 2
        
                for i in range(last_index):
                    entire_array[k][i], entire_array[k][last_index + i] = entire_array[k][last_index + i], entire_array[k][i]

    return entire_array
    
def changeScoreToBinary(entire_array):
    #this is based on the score being present in the last two columns
    for i in range(len(entire_array)):
        #print(entire_array[i][-1],  entire_array[i][-2])
        if entire_array[i][-1] > entire_array[i][-2]:
            
            entire_array[i][-1] = 1
            entire_array[i][-2] = 0
        else:
            entire_array[i][-1] = 0
            entire_array[i][-2] = 1

    return entire_array



def trainModel(iterations, entire_set, train_size, load=None, save=None):
    
    train_set = np.array(entire_set[0:train_size, 0:num_features], dtype=np.float32)    
    x_gpu = cuda.to_gpu(train_set, device=0)
    
    target = np.array(entire_set[0:train_size, num_features:], dtype=np.int32)
    target_gpu = cuda.to_gpu(target[:,0], device=0)

    if load == "yes": serializers.load_npz('NFL boy v.03softmax 04', model)
    
    print("train set size", len(train_set))
    if iterations > 4:
        for i in range(iterations):        
            if i % 2 == 0:
                    entire_set = randomSwapping(entire_set)
                    #print("first few lines", entire_set[0:5,20:])
                    
                    train_set = np.array(entire_set[0:train_size, 0:num_features], dtype=np.float32)    
                    x_gpu = cuda.to_gpu(train_set, device=0)
    
                    target = np.array(entire_set[0:train_size, num_features:], dtype=np.int32)
                    target_gpu = cuda.to_gpu(target[:,0], device=0)
            
            
            #print(output.shape)
            #print(target.shape)
            
            model.cleargrads() 
            loss = model(x_gpu, target_gpu, train=True)
            loss[0].backward()
            #print('loss 1' , loss[0].data)

            optimizer.update()
            #if i % 1000 == 0:
            print("iteration", i,
                  'loss' , loss[0].data,
                  "accuracy", loss[1].data * 100)
            #print('accuracy', F.accuracy(x_gp, target_gpu))
            #print('output' , output.data)
            
            #print('loss' , loss.data)
            #print('output' , output.data)
    
    if save == "yes": serializers.save_npz('NFL boy v.03softmax 04', model)

    # validation
    #validationWinsGraph(entire_set)
    #generateMedianErrors(entire_set)


def validationIndividual(entire_set):
    
    
    correct_game_index = []
    correct_winner = 0
    incorrect_winner = 0
    entire_set = entire_set.astype('f')
    model.to_cpu()
    free_model.cleargrads()
    
    #target = entire_set[train_size:, num_features]
    targets = np.array(entire_set[train_size:, num_features:], dtype=np.int32)
    #print("target is", target)
    predictions = free_model(entire_set[train_size:, 0:num_features], train=False)
    
    output = F.softmax(predictions.data)

    values_to_delete = []

    for i in range(len(entire_set)):
        if abs(output[i][0].data.tolist() - output[i][1].data.tolist()) < .20:
            values_to_delete.append(i)
    
    print("number of values to delete", len(values_to_delete))        
    output = output.data.tolist()
    
    output = np.delete(output, values_to_delete, 0)
    targets = np.delete(targets, values_to_delete, 0)     
    plot = np.zeros((len(output),2))
    print("lenght of predictions", len(output), "len of targ", len(targets))
    if len(output > 0):
        for i in range(len(output)):
            #print("output", output[i][0].data.tolist(), output[i][1].data.tolist())
            #print("target", targets[i][0])
            if ((output[i][0] >= output[i][1]) and (targets[i][0] >= 1)):
                correct_winner += 1
                correct_game_index.append(i)
                #print("correct")
            elif ((output[i][0] < output[i][1]) and (targets[i][0] < 1)):
                correct_game_index.append(i)
                correct_winner += 1 
                #print("correct")
            else:
                incorrect_winner += 1
                #print("incorrect")
            plot[i,0] = correct_winner
            plot[i,1] = incorrect_winner
    
        print("correct", correct_winner,
              "incorrect", incorrect_winner,
              "accuracy", 1 - (correct_winner / float(correct_winner + incorrect_winner)),
              "total games", correct_winner + incorrect_winner)
    else:
        print("not enough games above threshold")
    return plot, correct_game_index, values_to_delete

    

def validationWinsGraph(entire_set):            
    correct_winner = 0
    incorrect_winner = 0
    entire_set = entire_set.astype('f')
    model.to_cpu()
    model.cleargrads()
    #target = entire_set[train_size:, num_features]
    target = np.array(entire_set[train_size:, num_features], dtype=np.int32)
    
    predictions = model(entire_set[train_size:, 0:num_features], target)  
    print("###########################")
    print("loss,", predictions[0].data)
    print("accuracy,", predictions[1].data)
    
    print("correct winner:", len(target) * predictions[1].data)  
    print("incorrect winner:", len(target) - (len(target) * predictions[1].data))

    return predictions

def deleteColumns(array, columns):
    for column in columns:
        del array[column]
    return array

def reverseRandom(changed_scores, predictions):
    indexes = np.where(changed_scores[:,0] < changed_scores[:,1])
    indexes = indexes[0] #get rid of that damnf irst dimsenion
    predictions[indexes,0], predictions[indexes,1] = \
        predictions[indexes,1], predictions[indexes,0]
    
    df = pd.DataFrame(predictions)
    df.to_csv("2016 NFL prob predictions 01.csv")

    #return predictions #winners on left, losers on right


columns_to_delete = ["Team A", "Team B", "Unnamed: 0", "Week",
                     "Day", "Time", "Date", "Home"]

training_set = pd.read_csv("Datasets/Training Data/NFL training set 2.csv")
#training_set = pd.read_csv("NFL 2015 training data.csv")

training_set = deleteColumns(training_set, columns_to_delete)
training_set_np = training_set.values
training_set_np = randomSwapping(changeScoreToBinary(training_set_np))
training_set_np = np.nan_to_num(training_set_np)


trainModel(ITERATIONS, training_set_np, train_size, load=LOAD, save=SAVE)


validation_set = pd.read_csv(("Datasets/Training Data/NFL validation set 2.csv"))
validation_set = deleteColumns(validation_set, columns_to_delete) 
validation_set_np = validation_set.values
validation_set_np = np.nan_to_num(validation_set_np)
validation_set_np = randomSwapping(randomSwapping(changeScoreToBinary(validation_set_np)))

train_size = 0
print("2016 games")
validationWinsGraph(validation_set_np)



plt.title("Validation Set: Correct vs Incorrect Win Estimate")
plt.ylabel('# of games right/wrong')
plt.xlabel("Total number of games")


red_patch = mpatches.Patch(color='red', label='Incorrect')
blue_patch = mpatches.Patch(color='blue', label='Correct')

plt.legend(handles=[red_patch, blue_patch], loc = 'upper left')



plot, correct_list, under_threshold_game_index = validationIndividual(validation_set_np)
#plot_2 = validationIndividual(validation_set_np_2)

t = np.arange(0, len(plot), 1)
#g = np.arange(0, len(plot_2),1)
plt.plot(t, plot[:,1], 'b', t,plot[:,0], 'r') 
#plt.plot(g, plot_2[:,1], 'black', g, plot_2[:,0], 'g',)

plt.show()