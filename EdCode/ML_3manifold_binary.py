'''ML 3-manifold strings'''
'''to do: more data, more hyperparam optimisation?'''
'''to run:
    ~ ensure filepath is correct for data and listed libraries installed
    ~ set the NN hyperparameters at the start of cell 4 - options to scale the data with boolean 'scale', and to run a single (True) or SNN (False) architecture with boolean 'preconcat' indicating if the two strings are preconcatenated before input into the NN
    ~ run the cells sequentially (choosing either cells 2 or 3 but not both) to respectively: import libraries, import data, set up data, train ML models, test ML models
'''
## Cell 1 ## --> Import libraries
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import csv
from collections import Counter
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
#from multiprocess import Pool #...models cant be pickled to be passed to processes

#%% ## Cell 2 ## --> Import Data
with open('./Data/manifold_data.csv', newline='') as f: #...ensure filepath correct
    reader = csv.reader(f)
    data = list(reader)

#Identify vocabulary of all unique letters
all_strings = np.array(data,dtype='str')[:,:2].flatten()
max_strlen = max(map(len,all_strings))
LetterDist = Counter(''.join([i for i in all_strings]))

#Vocab listed in order extracted
vocab =list(LetterDist.keys()) 
#Vocab listed in descending order
vocab = sorted([list(i) for i in LetterDist.items()], key=lambda x: x[1], reverse=True)

spread_vocab = False #...select whether to reorganise the vocab -> index allocation so not in descending order (performance seems to be worse with this? set to True to try with)
#Sort vocab so as to spread similar frequency letters across the range ...comments: most or least common letters more intersting? Want similar frequency letters to be allocated different integers i think? 
if spread_vocab:
    #Define function to compute measure of the vocab ordering    
    def vocab_weight(vocab_order): #...compute the weight of a vocabulary order according to: \sum_i [freq_i * \sum_{j \neq i}(\frac{1}{|letterindex_i-letterindex_j|}*freq_j)]
        measure = 0
        for l_idx in range(len(vocab_order)):
            update = 0
            for l_idx2 in list(range(l_idx))+list(range(l_idx+1,len(vocab_order))):
                update += 1/np.absolute(l_idx2-l_idx) * vocab_order[l_idx2][1] * 1e-8 #...added a scaling to keep the values in the float range
            measure += vocab_order[l_idx][1] * update
        return measure
    
    #Sample so many permutations and choose the best
    current_measure = vocab_weight(vocab)
    print('OG:',current_measure)
    for perm in range(1000):
        trail_perm = shuffle(vocab)
        new_measure = vocab_weight(trail_perm)
        if new_measure < current_measure:
            current_measure = new_measure
            best_vocab = trail_perm
            print(perm+1,new_measure)
            
    vocab = [x[0] for x in best_vocab]
    
else: vocab = [x[0] for x in vocab] 

#Convert strings to vectors and correct other types
for pair in data:
    word1, word2, = [], []
    for letter in pair[0]:
        word1.append(1*(vocab.index(letter)+1)) #...the 1* is present as trialled scaling the integere assignments, but learning not improved; left in if one wishes to play with
    for letter in pair[1]:
        word2.append(1*(vocab.index(letter)+1)) #...the 1* is present as trialled scaling the integere assignments, but learning not improved; left in if one wishes to play with
    pair[0], pair[1] = np.array(word1), np.array(word2)
    pair[2]=float(int(pair[2]))
    pair[3]=bool(pair[3])
del(reader,f,pair,word1,word2,letter)

#%% ## Cell 3 ## --> Import easier data (run instead of above cell and unhash the line for the dataset interested in)
#with open('./Data/early_closed_data.csv', newline='') as f: #...ensure filepath correct
with open('./Data/early_cusped_data.csv', newline='') as f: #...ensure filepath correct
    reader = csv.reader(f)
    data = list(reader)
#Identify vocabulary of all unique letters
all_strings = np.array(data,dtype='str')[:,4:6].flatten()
max_strlen = max(map(len,all_strings))
LetterDist = Counter(''.join([i for i in all_strings]))

#Extract vocab
vocab =list(LetterDist.keys()) 
vocab = sorted([list(i) for i in LetterDist.items()], key=lambda x: x[1], reverse=True)
vocab = [x[0] for x in vocab] 

#Convert strings to vectors and correct other types
newdata=[]
for pair in data:
    newdata.append([])
    word1, word2, = [], []
    for letter in pair[4]:
        word1.append(1*(vocab.index(letter)+1))
    for letter in pair[5]:
        word2.append(1*(vocab.index(letter)+1))
    newdata[-1].append(np.array(word1))
    newdata[-1].append(np.array(word2))
    newdata[-1].append(float(int(pair[6])))
data = newdata
del(reader,f,pair,word1,word2,letter,newdata)

#%% ## Cell 4 ## --> Define NN hyper-parameters & set-up data
k_cv=5                  #...number of cross-validations to run
split=0.2               #...if only 1 run (k_cv=1) then proportion of test data
bs=64                   #...batch size to use in each training step
num_epochs=50           #...number of training epochs to run
lr=1e-4                 #...define optimizer learning rate
dp=0.01                 #...set to non-zero to include dropout layers with that rate
Pad_choice=0            #...currently always pad 
preconcat = False       #...option to learn with concatenating pairs before training (True), or within the model (False)
scale = False           #...option to standardise the input dataset scaling (True)
if preconcat: layer_sizes = [256,64,32]     #...number and size of the dense NN layers for the TrainNN function, with format [a,b,c]
else:         layer_sizes = [[256,64],[32]] #...number and size of the dense NN embedding layers and then the combined layers for the TrainDualNN function, with format [[a,b,c],[d,e]]

#Set-up data
if preconcat:
    #Set-up X (input) & Y (output) data (including padding)
    X = np.array([np.pad(np.concatenate((pair[0],pair[1])),(0,2*max_strlen-(len(pair[0])+len(pair[1]))),constant_values=Pad_choice) for pair in data])
    Y = np.array([pair[2] for pair in data])
    #Shuffle the data
    X, Y = shuffle(X, Y)
    #Scale the data
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    #Perform train:test split (currently doing validation implicitly within .fit() but could hardcode if want)
    if k_cv == 1:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split, random_state=None, shuffle=True)
        X_train, X_test, Y_train, Y_test = [X_train], [X_test], [Y_train], [Y_test]
        #X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=None, shuffle=True) #...do validation implicitly within model.fit()
    elif k_cv > 1:
        X_train, Y_train, X_test, Y_test = [], [], [], []
        s = int(floor(len(data)/k_cv)) #...number of datapoints in train split
        for i in range(k_cv):
            X_train.append(np.concatenate((X[:i*s],X[(i+1)*s:])))
            Y_train.append(np.concatenate((Y[:i*s],Y[(i+1)*s:])))
            X_test.append(X[i*s:(i+1)*s])
            Y_test.append(Y[i*s:(i+1)*s])
else: 
    #Set-up Xa&Xb (input) & Y (output) data (including padding)
    Xa = np.array([np.pad(pair[0],(0,max_strlen-len(pair[0])),constant_values=Pad_choice) for pair in data])
    Xb = np.array([np.pad(pair[1],(0,max_strlen-len(pair[1])),constant_values=Pad_choice) for pair in data])
    Y = np.array([pair[2] for pair in data])    
    #Shuffle the data
    Xa, Xb, Y = shuffle(Xa, Xb, Y)
    #Scale the data
    if scale:
        scaler = StandardScaler()
        scaler.fit(np.concatenate((Xa,Xb)))
        Xa = scaler.transform(Xa)
        Xb = scaler.transform(Xb)
    
    #Perform train:test split (currently doing validation implicitly within .fit() but could hardcode if want)
    if k_cv == 1:
        Xa_train, Xa_test, Xb_train, Xb_test, Y_train, Y_test = train_test_split(Xa, Xb, Y, test_size = split, random_state=None, shuffle=True)
        Xa_train, Xa_test, Xb_train, Xb_test, Y_train, Y_test = [Xa_train], [Xa_test], [Xb_train], [Xb_test], [Y_train], [Y_test]
        #Xa_train, Xa_val,  Xb_train, Xb_val,  Y_train, Y_val = train_test_split(Xa_train, Xb_train, Y_train, test_size = 0.1, random_state=None, shuffle=True) #...do validation implicitly within model.fit()
    elif k_cv > 1:
        Xa_train, Xa_test, Xb_train, Xb_test, Y_train, Y_test = [], [], [], [], [], []
        s = int(floor(len(data)/k_cv)) #...number of datapoints in train split
        for i in range(k_cv):
            Xa_train.append(np.concatenate((Xa[:i*s],Xa[(i+1)*s:])))
            Xb_train.append(np.concatenate((Xb[:i*s],Xb[(i+1)*s:])))
            Y_train.append(np.concatenate((Y[:i*s],Y[(i+1)*s:])))
            Xa_test.append(Xa[i*s:(i+1)*s])
            Xb_test.append(Xb[i*s:(i+1)*s])
            Y_test.append(Y[i*s:(i+1)*s])    
    
#%% ## Cell 5 ## --> Run ML training
#Define the neuron activation function to use
def act_fn(x): return keras.activations.relu(x,alpha=0.01) #...leaky-ReLU activation

#Define NN function (build & train)
def TrainNN(layer_sizes,train_data,test_data):
    #Setup NN
    model = keras.Sequential()
    for layer_size in layer_sizes:
        model.add(keras.layers.Dense(layer_size, activation=act_fn))
        if dp: model.add(keras.layers.Dropout(dp)) #...dropout layer to reduce chance of overfitting to training data
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics='accuracy') 
    #Train NN
    history = model.fit(train_data, test_data, batch_size=bs, epochs=num_epochs, shuffle=True, validation_split=0.1, verbose=0, use_multiprocessing=1, workers=4)
    return model, history
    
def TrainDualNN(layer_sizes,train_data,test_data): 
    #Setup NN
    w1_embed = keras.models.Sequential()
    w2_embed = keras.models.Sequential()
    w1_embed.add(keras.layers.InputLayer(input_shape=train_data[0].shape[1]))
    w2_embed.add(keras.layers.InputLayer(input_shape=train_data[1].shape[1]))
    for layer_size in layer_sizes[0]:
        w1_embed.add(keras.layers.Dense(layer_size, activation=act_fn))
        w2_embed.add(keras.layers.Dense(layer_size, activation=act_fn))
        if dp: w1_embed.add(keras.layers.Dropout(dp)) #...dropout layer to reduce chance of overfitting to training data
        if dp: w2_embed.add(keras.layers.Dropout(dp)) #...dropout layer to reduce chance of overfitting to training data
    w1w2_combined = keras.layers.concatenate([w1_embed.output,w2_embed.output])
    for layer_size in layer_sizes[1]:
        w1w2_combined = keras.layers.Dense(layer_size, activation=act_fn)(w1w2_combined)
        if dp: w1w2_combined = keras.layers.Dropout(dp)(w1w2_combined) #...dropout layer to reduce chance of overfitting to training data
    w1w2_combined = keras.layers.Dense(1, activation='sigmoid')(w1w2_combined)
    #Compile NN
    model = keras.models.Model(inputs=[w1_embed.input,w2_embed.input], outputs=w1w2_combined)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics='accuracy') 
    #Train NN
    history = model.fit(train_data, test_data, batch_size=bs, epochs=num_epochs, shuffle=True, validation_split=0.1, verbose=0, use_multiprocessing=1, workers=4)
    return model, history

#Run NN training
models, histories = [], []
for run in range(k_cv):
    if preconcat:   m, h = TrainNN(layer_sizes,X_train[run],Y_train[run]) 
    else:           m, h = TrainDualNN(layer_sizes,[Xa_train[run],Xb_train[run]],Y_train[run]) 
    models.append(m)
    histories.append(h)
    print('Final validation accuracy: '+str(h.history['val_accuracy'][-1]))
del(run,m,h)

#Pool the training #...can't pickle the models to pass to processes
#with Pool() as p:
    #models, histories = p.starmap(TrainNN, [(layer_sizes,X_train[i],Y_train[i]) for i in range(k_cv)])

##%% #Training plots
plt.figure()
#for run in range(k_cv): #...to plot all runs individually
    #plt.plot(histories[run].history['accuracy'],label='Run '+str(run)+' acc')
    #plt.plot(histories[run].history['val_accuracy'],label='Run '+str(run)+' val-acc') 
plt.plot(range(1,num_epochs+1),np.mean(np.array([run.history['accuracy'] for run in histories]),axis=0),label='acc')
plt.plot(range(1,num_epochs+1),np.mean(np.array([run.history['val_accuracy'] for run in histories]),axis=0),label='val-acc')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xticks(range(0,num_epochs+1,5))
plt.xlim(0,num_epochs+1)
plt.ylim(0,1)
plt.legend(loc='best')
plt.grid()
plt.show()
    
#%% ## Cell 6 ## --> Run NN testing
accuracies, mccs, cms = [], [], []
for run in range(k_cv):
    if not preconcat: p = np.round(models[run].predict([Xa_test[run],Xb_test[run]]))
    else: p = np.round(models[run].predict(X_test[run]))
    accuracies.append(accuracy_score(Y_test[run],p))
    mccs.append(matthews_corrcoef(Y_test[run],p))
    cms.append(confusion_matrix(Y_test[run],p,normalize='true'))
del(run,p)

#Print averaged results
print('Averaged Learning Measures:')
print('Accuracy:\t'+str(np.mean(accuracies))+'\t$\pm$ '+str(np.std(accuracies)/k_cv))
print('MCC:\t\t\t'+str(np.mean(mccs))+'\t$\pm$ '+str(np.std(mccs)/k_cv))
print('CM:\n'+str(np.mean(cms,0))+'\n$\pm$\n'+str(np.std(cms,0)/k_cv))

