import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.argv[1] 
import numpy as np

def plot_data(fileName):
    data = pd.read_csv(fileName)
    
    plt.plot(data['year'], data['days'])
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig("plot.jpg") 
    
    return

def linear_regression(fileName):
    
    data = pd.read_csv(fileName)

    # Matrix X
    X = np.c_[np.ones(len(data)), data['year'].values]  

    # Vector Y
    Y = data['days'].values.reshape(-1, 1)
    
    # Matrix Product Z = X^T X
    Z = np.dot(X.T, X)
    
    # Inverse of X^T * X
    I = np.linalg.inv(Z)
    
    # Pseudo-Inverse of X
    PI = np.dot(I, X.T)
    
    # Regression Coefficients  beta hat
    beta_hat = np.dot(PI, Y)
    
    return X, Y, Z, I, PI, beta_hat


def predict(beta_hat, x_test):
    
    y_test = beta_hat[0] + beta_hat[1] * x_test
    
    return y_test


if __name__ == "__main__":
    
    fileName = sys.argv[1]
    
#     2. 
    plot_data(fileName)
    
    
#     3.
    X, Y, Z, I, PI, beta_hat = linear_regression(fileName)
    
    print("Q3a:")
    print(X.astype('int64')) 
    
    print("Q3b:")
    print(Y.astype('int64').flatten())  
    
    print("Q3c:")
    print(Z.astype('int64')) 
    
    print("Q3d:")
    print(I)  
    
    print("Q3e:")
    print(PI) 
    
    print("Q3f:")
    print(beta_hat.flatten()) 
    
    
#     4. 
    x_test=2022
    y_test= predict(beta_hat, x_test)
    
    print("Q4: " + str(np.squeeze(y_test)))
    
    
#    5. 
    beta_1 = beta_hat[1]
    
    if beta_1 > 0:
        symbol = ">"
    elif beta_1 < 0:
        symbol = "<"
    else:
        symbol = "="
    
#     5a. 
    print("Q5a: ", symbol)
    
    answer = "If > is outputted: beta 1 hat is posotive. this indicates that their is a posotive correlation between the year and the number of ice days on Lake Mendota. As the year increases, and time passes, the number of ice days tends to increase as well.      If < is outputted: beta 1 hat is negative. This indicates that their is a negative correlation between the year and the number of ice days on Lake Mendota. As the year increases, and time passes, the number of ice days tends to decrease. If = is outputted: beta 1 hat is 0. This indicates that their is no correlation between the year and the number of ice days on lake mendota."
#     5b. 
    print("Q5b: ", answer) 
    
    
#     6. 

    x_star = -beta_hat[0] / beta_hat[1] 
    
#     6a. 
    print("Q6a: ", np.squeeze(x_star))
    
#     6b. 


    print("Q6b: This suggests that lake mendota will stop freezing in the year 2456. This is a reasonable prediction given the trend in the data, since the trends indicate that the freeze days will continually decrease over time until we reach a point where there a zero freeze days. However, i disagree with this prediction because it does not take in to account the long-term effects of climate change, or if some extreme natural disater occurs within the next 300 years that impact the rate climate change. There is also the possibility that future technology is able to prevent/decrease the impacts of climate change, leading to x* being much later. ")


    
    



    
    