
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):
    
    data=[]
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
            
    
    return data


def calc_features(row):
    
    
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])
    
    features = np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)
    return features


def hac(features):
    n = len(features)

    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if(i==j):
                distance_matrix[i][j]=np.inf
            else:
                distance_matrix[i][j] =np.linalg.norm(features[i] - features[j])

                
    clusters = {i: [i] for i in range(n)}
    

        
    Z = np.zeros((n - 1, 4))
    cluster_matrix = np.full( (2*n-1,2*n-1) , np.inf)
    for i in range(n):
        for j in range(n):
            if i ==j :
                cluster_matrix[i][j] = np.inf
            else: 
                cluster_matrix[i][j]= distance_matrix[i][j]
               
    
             
    for i in range(n-1):
        index = np.argmin(cluster_matrix)
        r,c = np.unravel_index(index, cluster_matrix.shape)
        Z[i, 0] = r
        Z[i, 1] = c
        Z[i, 2] = cluster_matrix[r][c]
        Z[i, 3] = len(clusters[r]) + len(clusters[c])
        
        clusters[n+i] = clusters[r] + clusters[c]
        clusters[r] = []
        clusters[c] = []

        
        for j in range(n+i):

            cluster_matrix[n+i][j]= helper_hac(n+i, j, clusters, distance_matrix)
            cluster_matrix[j][n+i]=cluster_matrix[n+i][j]
        
      
        cluster_matrix[r]=np.inf
        cluster_matrix[:, c] = np.inf
        cluster_matrix[c]=np.inf
        cluster_matrix[:, r] = np.inf
        
    return Z


def helper_hac(cluster1, cluster2, clusters, cluster_matrix):
    list1= clusters[cluster1]
    list2= clusters[cluster2]
    max_dist= 0
    
    if(len(list1) == 0 or len(list2) == 0):
        return np.inf
    
    for i in list1:
        for j in list2:
            if cluster_matrix[i][j]>max_dist:
                max_dist = cluster_matrix[i][j]
                
    return max_dist
                
        

def fig_hac(Z, names):
    fig = plt.figure(figsize=(10, 6)) 

    dendrogram(Z, labels=names, leaf_rotation=90)  

    plt.xlabel('Countries')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering Dendrogram')

    plt.tight_layout()

    
    return fig
    
    
def normalize_features(features):
    
    features = np.array(features)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    normalized_features = []
    for feature_vector in features:
        normalized_feature_vector = (feature_vector - means) / stds
        normalized_features.append(normalized_feature_vector)
    return normalized_features
    
    
    
    