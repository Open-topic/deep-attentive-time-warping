import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np
import logging

log = logging.getLogger(__name__)

def mds_visualization(alldist,train_label,test_label,predict):
    # Create DataFrame
    df = pd.DataFrame(alldist, columns=train_label, index=test_label)

    df['predict'] = predict

    df.columns = df.columns.astype(str)
    
    
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
    filepath = '/content/drive/MyDrive/deep-attentive-time-warping/mds_visualization_dtw.png'
    plt.savefig(filepath,dpi=300)

    print("Image saved at:", filepath)

    # Display scatterplot
    plt.show()

