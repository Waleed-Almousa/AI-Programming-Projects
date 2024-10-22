from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x=np.load(filename)
    x= x - np.mean(x, axis=0)
    return x
    
    
def get_covariance(dataset):
    n = dataset.shape[0]
    covariance =  np.dot(np.transpose(dataset), dataset)/(n-1)
    return covariance
    

def get_eig(S, m):

    eigenvalues, eigenvectors = eigh(S, subset_by_index=[S.shape[0]-m, S.shape[0]-1])
    
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    m_eigenvalues = np.diag(eigenvalues[:m])
    m_eigenvectors = eigenvectors[:, :m]
    
    return m_eigenvalues, m_eigenvectors
        

def get_eig_prop(S, prop):
            
    eigenvalues, eigenvectors = eigh(S)
    
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    total_variance = np.sum(eigenvalues)
    
    cumulative_variance = np.cumsum(eigenvalues) / total_variance
    
    idx = np.argmax(cumulative_variance > prop)
    
    selected_eigenvalues = eigenvalues[:idx+1]
    selected_eigenvectors = eigenvectors[:, :idx+1]
    

    selected_diagonal = np.diag(selected_eigenvalues)
    
    return selected_diagonal, selected_eigenvectors
    

def project_image(image, U):
    
    projection = np.dot(U.T, image)
    
    return np.dot(U, projection)

def display_image(orig, proj):
    
    orig = orig.reshape(64, 64)
    orig=orig.T
    proj=proj.reshape(64, 64)
    proj=proj.T
   
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    
    ax1.set_title("Original")
    ax2.set_title("Projection")

    im1= ax1.imshow(orig, aspect='equal')
    cbar1 = fig.colorbar(im1, ax=ax1)


    im2 = ax2.imshow(proj, aspect='equal')
    cbar2 = fig.colorbar(im2, ax=ax2)
    
    
    return fig, ax1, ax2

















