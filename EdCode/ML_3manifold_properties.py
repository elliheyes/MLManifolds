'''ML 3-manifold properties from strings'''
'''to do: hyperparam optimisation'''
'''to run: 
    ~ ensure filepath is correct for data and listed libraries installed
    ~ select the parameter to learn using {0,1,2,3} at start of cell 2
    ~ set the NN hyperparameters at the start of cell 3
    ~ run cells sequentially to respectively: import libraries, import data, set up data, train ML models, test ML models, plot data histograms
'''
## Cell 1 ## --> Import libraries
import numpy as np
import matplotlib.pyplot as plt
import csv
from math import floor
from ast import literal_eval
from collections import Counter
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#%% ## Cell 2 ## --> Import Data
#Select which parameter to learn (0: volume, 1: homotopy, 2: geodesics, 3: tetrahedra)
param = 0

#Open and read database
with open('./Data/NEWclosedmanifolddatabase3000.csv', newline='') as f: #...ensure filepath correct
    reader = csv.reader(f)
    rawdata = list(reader)
    
#Extract a string and the volume for each manifold
data = []
for row in rawdata:
    #Volume - predict volume
    if param == 0:
        data.append([literal_eval(row[4])[1],float(row[2])]) 
        #Extract all strings per manifold - currently worse with more data (cause the triangulations v different?)
        #for string in row[4:]: 
            #data.append([literal_eval(string)[1],float(row[2])]) 
    #Homotopy - predict quotients of the homotopy group
    elif param == 1:
        hmtpy = []
        for grp in row[1].replace(' ','').split('+'):   #...data is list of all Z quotients in sum ('0' taken as '0')
            if grp.split('/')[-1]=='Z': hmtpy.append(1) #...just 'Z' has no integer to interpret so handle separately
            else: hmtpy.append(int(grp.split('/')[-1])) #...otherwise just take the integer quotient
        data.append([literal_eval(row[4])[1],hmtpy])
    #Geodesics - predict the number of shortest geodesics
    elif param == 2:
        data.append([literal_eval(row[4])[1],sum([int(g[0]) for g in (row[3]).split('\n')[1:]])])        
    #Tetrahedra - predict number of tetrahedra of the isomorphic signature string
    elif param == 3:
        for string in row[4:]: 
            data.append([literal_eval(string)[1],int(literal_eval(string)[0])]) 
        
#Identify vocabulary of all unique letters
all_strings = np.array([i[0] for i in data],dtype='str').flatten() 
max_strlen = max(map(len,all_strings))
LetterDist = Counter(''.join([i for i in all_strings]))
vocab = list(LetterDist.keys())

#Convert strings to vectors and correct other types
for manifold in data:
    word = []
    for letter in manifold[0]:
        word.append(vocab.index(letter)+1)
    manifold[0] = np.array(word)
    if param == 1: manifold[1]=np.pad(np.array(manifold[1]),(0,max(map(len,[i[1] for i in data]))-len(manifold[1])),constant_values=0) #...pad quotients lists to max length with zeros
del(reader,f,row,manifold,word,letter)
if param == 1: del(hmtpy,grp)
elif param == 3: del(string)

#%% ## Cell 3 ## --> Define NN hyper-parameters   & set-up data  
k_cv=5                  #...number of cross-validations to run
split=0.2               #...if only 1 run (k_cv=1) then proportion of test data
bs=64                   #...batch size to use in each training step
num_epochs=30           #...number of training epochs to run
lr=1e-4                 #...define optimizer learning rate
dp=0.01                 #...set to non-zero to include dropout layers with that rate, or 0 for no dropout
Pad_choice=0            #...currently always pad 
layer_sizes = [512,128,32] #...number and size of the dense NN layers (for TrainNN format [a,b,c], for TrainDualNN format [[a,b,c],[d,e]])

#Set-up X (input) & Y (output) data (including padding)
X = np.array([np.pad(manifold[0],(0,max_strlen-len(manifold[0])),constant_values=Pad_choice) for manifold in data])
Y = np.array([manifold[1] for manifold in data])

#Perform train:test split (currently doing validation implicitly within .fit() but could hardcode if want)
if k_cv == 1:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split, random_state=None, shuffle=True)
    X_train, X_test, Y_train, Y_test = [X_train], [X_test], [Y_train], [Y_test]
    #X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=None, shuffle=True) #...if wish to do validation explicitly (but then need to make an input to model.fit())
elif k_cv > 1:
    X_train, Y_train, X_test, Y_test = [], [], [], []
    s = int(floor(len(data)/k_cv)) #...number of datapoints in train split
    for i in range(k_cv):
        X_train.append(np.concatenate((X[:i*s],X[(i+1)*s:])))
        Y_train.append(np.concatenate((Y[:i*s],Y[(i+1)*s:])))
        X_test.append(X[i*s:(i+1)*s])
        Y_test.append(Y[i*s:(i+1)*s])
            
#%% ## Cell 4 ## --> Run ML training
#Define the neuron activation function to use
def act_fn(x): return keras.activations.relu(x,alpha=0.01) #...leaky-ReLU activation

#Define NN function (build & train)
def TrainNN(layer_sizes,train_data,test_data):
    #Setup NN
    model = keras.Sequential()
    for layer_size in layer_sizes:
        model.add(keras.layers.Dense(layer_size, activation=act_fn))
        if dp: model.add(keras.layers.Dropout(dp)) #...dropout layer to reduce chance of overfitting to training data
    if param in [0,2,3]:   model.add(keras.layers.Dense(1))
    elif param == 1:       model.add(keras.layers.Dense(max(map(len,[i[1] for i in data]))))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='logcosh') 
    #Train NN
    history = model.fit(train_data, test_data, batch_size=bs, epochs=num_epochs, shuffle=True, validation_split=0.1, verbose=0, use_multiprocessing=1, workers=4)
    return model, history

#Run NN training
models, histories = [], []
for run in range(k_cv):
    m, h = TrainNN(layer_sizes,X_train[run],Y_train[run]) 
    models.append(m)
    histories.append(h)
    print('Final Loss: '+str(h.history['val_loss'][-1]))
del(run,m,h)

##%% #Training plots
plt.figure()
#for run in range(k_cv): #...to plot all runs individually
    #plt.plot(histories[run].history['loss'],label='Run '+str(run)+' acc')
    #plt.plot(histories[run].history['val_loss'],label='Run '+str(run)+' val-acc') 
plt.plot(range(1,num_epochs+1),np.mean(np.array([run.history['loss'] for run in histories]),axis=0),label='loss')
plt.plot(range(1,num_epochs+1),np.mean(np.array([run.history['val_loss'] for run in histories]),axis=0),label='val loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xticks(range(0,num_epochs+1,5))
plt.xlim(0,num_epochs+1)
plt.ylim(0)
plt.legend(loc='best')
plt.grid()
plt.show()
    
#%% ## Cell 5 ## --> Run NN testing
acc_bound=0.5
mae, mse, acc, r2 = [], [], [], []
for run in range(k_cv):
    p = models[run].predict(X_test[run])
    mae.append(mean_absolute_error(Y_test[run],p))
    mse.append(mean_squared_error(Y_test[run],p))
    r2.append(r2_score(Y_test[run],p))
    if param in [0,2,3]:  acc.append((np.absolute(Y_test[run]-p.flatten())<acc_bound).sum()/len(Y_test[run]))
    elif param == 1:      acc.append((np.sum(np.absolute(Y_test[0]-p),axis=1)<acc_bound).sum()/len(Y_test[run]))
del(run,p)

#Print averaged results
print('Averaged Learning Measures:')
print('R^2:\t\t'+str(np.mean(r2))+'\t$\pm$\t'+str(np.std(r2)/k_cv))
print('MAE:\t\t'+str(np.mean(mae))+'\t$\pm$\t'+str(np.std(mae)/k_cv))
print('MSE:\t\t'+str(np.mean(mse))+'\t$\pm$\t'+str(np.std(mse)/k_cv))
print('Acc:\t\t'+str(np.mean(acc))+'\t$\pm$\t'+str(np.std(acc)/k_cv)+'\t\t...within bound: '+str(acc_bound))

#Volume range = 3.70794791821159, (min,max)=(0.98136882889,4.68931674710159)
#Homotopy range = 304, (min,max) = (0,304)
#Number geodesics range = 8, (min,max) = (0,8)
#Number tetrahedra range = 35, (min,max) = (9,44)
            
#%% ## Cell 1 ## --> Plot data histograms
if param == 0:
    #Plot histogram of volume data
    plt.hist(Y,bins=int(max(Y)*50),range=(0,max(Y)+0.25),histtype='step')
    plt.xlabel('Volume')
    plt.ylabel('Frequency')
    plt.xlim(0,5)
    #plt.ylim(0,500)
    plt.grid()
    plt.tight_layout()
    #plt.savefig('Volume_histogram.pdf')

elif param == 1:
    #Plot histogram of homotopy data
    plt.hist(Y.flatten(),bins=int(max(Y.flatten())*50),range=(0,max(Y.flatten())+5),histtype='step')
    plt.xlabel('Homotopy Quotient')
    plt.ylabel('Frequency')
    #plt.xlim(0,50)
    #plt.ylim(0,500)
    plt.grid()
    plt.tight_layout()
    #plt.savefig('HomotopyQuotient_histogram.pdf')
    
elif param == 2:
    #Plot histogram of number of shortest geodesics
    plt.hist(Y,bins=np.array(range(max(Y)+2))-0.5,histtype='step')
    plt.xlabel('Number of Geodesics')
    plt.ylabel('Frequency')
    #plt.xlim(-1,10)
    plt.grid()
    plt.tight_layout()
    #plt.savefig('NumberofGeodesics_histogram.pdf')
    print(Counter(Y))
    
elif param == 3:
    #Plot histogram of number of tetrahedra
    plt.hist(Y,bins=np.array(range(max(Y)+2))-0.5,histtype='step')
    plt.xlabel('Number of Tetrahedra')
    plt.ylabel('Frequency')
    plt.xlim(0)
    plt.grid()
    plt.tight_layout()
    #plt.savefig('NumberofTetrahedra_histogram.pdf')
    print(Counter(Y))

    
