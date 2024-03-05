# EE532L - Deep Learning for Healthcare - Programming Assignment 02
# Important: Please do not change/rename the existing function names and write your code only in the place where you are asked to do it.
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Write your code for the PCA below 
def my_PCA(X):
    x_mean=np.mean(X, axis=1)  
    deviation=X.T-x_mean
    cov=np.dot(deviation.T,deviation)
    eig_values,eig_vectors=np.linalg.eig(cov)
    indices=np.argsort(eig_values)
    eig_vectors=eig_vectors[indices]
  
    print(eig_vectors[:,:4].shape)
    data=np.dot(eig_vectors[:,:4].T,X)
    return(data)

def regress_fit(X_train, y_train, X_test):
    
    X_train = np.array(X_train)# Normalizing training features
    c = np.expand_dims(np.amax(X_train, axis = 1),axis=1)
    X_train = X_train / c
    
   
    
    X_test = np.array(X_test) # Normalizing testing features
    c = np.expand_dims(np.amax(X_test, axis = 1),axis=1)
    X_test = X_test / c
    
    p = X_train.shape[0] # Number of features
    N = X_train.shape[1] # Number of sample cases
    e = 0.00001 # Learning Rate (Hint:- adjust between 1e-5 to 1e-2)
    
    w = np.random.uniform(-1/np.sqrt(p), 1/np.sqrt(p), (p+1,1)) # Random initialization of weights
    X = np.ones((p+1,N)) # Adding an extra column of ones to adjust biases
    X[:p,:] = X_train
   
    
    # eps = 0.000000001 # infinitesimal element to avoid nan
    num_epochs = 2000
    
    loss,accuracy =[],[]
    for epoch in range(num_epochs): # Loop for iterative process
        J = 0 # Initializing loss
        count = 0 # Initializing count of correct predictions
        for i in range (N):
            z = ((w.T)@X[:,i:i+1])[0,0] # Raw logits (W.T x X)    
            
            # Q4) Write equation of Sigmoid(z)
            y = 1/(1+np.exp(-1*z)) # Sigmoid activation function
            T = y_train[i] # Ground Truth
            
            # Q5) Write loss function after the minus sign
            eps = 0.00000001
            J = J- T*np.log(y+eps)-(1-T)*np.log(1-y+eps)# Loss function
            # (Note:- The loss function is written after J = J- because we are trying to find
            # the average loss per epoch, so we need to sum it iteratively )
            
            # Q6) Write Derivative of J w.r.t z
            k = y-T # Derivative of J w.r.t z (Chain rule, J w.r.t y multiplied by y w.r.t z )
            dJ = k*X[:,i:i+1] # Final Derivative of J w.r.t w (dJ/dz multiplied by dz/dw)
            
            # Q7) Write formula of Gradient Descent
            w = w-e*dJ  # Gradient Descent
            
            if abs(y-T)<0.5 and T==1:
                count = count +1  # Counting the number of correct predictions
                
            
        
        
        
        train_loss = J/N
        train_accuracy = 100*count/N
        loss.append(train_loss)
        train_accuracy = 100*count/N
        accuracy.append(train_accuracy)

        batch_metrics = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} "
        sys.stdout.write('\r' + batch_metrics)
        sys.stdout.flush()
    
    plt.plot(range(1,num_epochs+1),loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(range(1,num_epochs+1),accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('acuracy')
    plt.show()

    
    # Testing
    print("\n")
    N2 = X_test.shape[1] # Number of test samples

    X2 = np.ones((p+1,N2)) # adding an additional columns of 1 to adjust biases
    X2[:p,:] = X_test

    z2 = w.T@X2 # test logit matrix
    y_pred = 1/(1+np.exp(-z2)) # Sigmoid activation function to convert into probabilities
    y_pred[y_pred>=0.5] = 1 # Thresholding
    y_pred[y_pred<0.5] = 0


        


    return y_pred
###########################################################################################################################################


########################################################## Cannot be modified ##############################################################
# Load the dataset
def load_and_fit():
    df = pd.read_csv("/content/drive/MyDrive/diabetes.csv")
    X = df.drop("Outcome", axis=1)

    X2 = np.array(X)
    np.random.seed(208)
    np.random.shuffle(X2)
    
    X2 = X2.T
    X2 = my_PCA(X2)
    y = df["Outcome"]
    X_train = X2[:,:614]
    
    X_test = X2[:,614:]
    y_train = y[:614]
    
    y_test = y[614:]
    
    

    # Fit the model
    y_test_pred = regress_fit(X_train, y_train, X_test)

    # Evaluate the accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred[0])
    print(f"Test Accuracy: {test_accuracy:.5f}")
    return round(test_accuracy, 5)

a = load_and_fit()
