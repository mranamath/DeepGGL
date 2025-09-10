import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from argparse import RawDescriptionHelpFormatter
import argparse

from scipy import stats
import sys
sys.path.append('../training')
from train import *

if __name__ == "__main__":

    d = """
        Predict protein-ligand binding affinity.
        """
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('-tf', '--test_feature', help='file path of test feature')
    parser.add_argument("-scaler", type=str,
                        help="training scaler file for preprocessing the features.")

    parser.add_argument("-model", type=str, default="DeepGGL.h5",
                        help="saved best model")

    parser.add_argument("-out", type=str, default="prediction.csv",
                        help="output prediction file name")

    args = parser.parse_args()

    # Get test data
    df_test = pd.read_csv(args.test_feature)

    ytest = df_test['pK'].values
    pdbids = df_test['PDBID'].values

    Xtest = df_test.drop(['PDBID','pK'], axis=1)
    Xtest = Xtest.values

    del df_test
    
    scaler = joblib.load(args.scaler)
    Xtest = scaler.transform(Xtest).reshape(-1, 74, 112, 3)
    
    # Load the model with custom object 'Weighted_PCC_RMSE'
    model = tf.keras.models.load_model(args.model, 
                                       custom_objects={'Weighted_PCC_RMSE': Weighted_PCC_RMSE})
   
    pred_pKa = model.predict([Xtest]).ravel()

    pred_df = pd.DataFrame(data={'PDBID':pdbids,
                            'Exp pK': ytest,
                            'Predicted pK': pred_pKa})
    pred_df.to_csv(args.out, index=False, float_format="%.3f")



