import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np
import logging

log = logging.getLogger(__name__)

def mds_visualization(alldist,train_label,test_label,predict):
    # Create DataFrame
    df = pd.DataFrame(alldist, columns=train_label, index=test_label)

    # add predicted_label column to df 
    df['predict'] = predict

    df.columns = df.columns.astype(str)
    
    # Save DataFrame to a CSV file
    df.to_csv('/content/drive/MyDrive/deep-attentive-time-warping/data_input_to_MDS.csv', index=True)
    
    # Perform multi-dimensional scaling
    mds = MDS(random_state=0)
    X_reduced = mds.fit_transform(df)

    # Visualize the reduced data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=test_label, cmap=plt.cm.get_cmap("jet", int(np.amax(test_label)) ) )
    plt.colorbar(label='Test_data Label',ticks=range(int(np.amax(test_label))+1) )
    plt.title("MDS Visualization")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    
    #save image
    filepath = '/content/drive/MyDrive/deep-attentive-time-warping/mds_visualization.png'
    plt.savefig(filepath,dpi=300)

    print("Image saved at:", filepath)

    # Display scatterplot
    plt.show()

