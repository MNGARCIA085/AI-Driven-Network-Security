import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig
from src.preprocessors.factory import PreprocessorFactory
import pandas as pd





import mlflow
import mlflow.pytorch






mlflow.set_tracking_uri("sqlite:///mlflow.db")







@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    model_name = cfg.model_type





    ##########################################################
    #######to simulate an scaler and an encoder###############
    ##########################################################

    # get preprocessor    
    preprocessor = PreprocessorFactory.get_preprocessor(model_name, cfg, cfg.preprocessor) # e.g. NNPreprocessor

    # preprocess
    data = preprocessor.preprocess() 
    X_train, X_val, y_train, y_val, artifacts = data
    
    aux = preprocessor.get_artifacts()
    scaler = aux['scaler']
    encoder = aux['encoder']

    #print(scaler.mean_)
    #print(encoder)

    print(X_train)



    ##########################################################
    ###################   TEST ###############################
    ##########################################################



    # load data
    path = "data/test_data.csv"
    df = pd.read_csv(path)


    # I need scaler and encoder (later they will be the corresponding to the best run)



    #
    preprocessor2 = PreprocessorFactory.get_preprocessor(model_name, cfg, cfg.preprocessor)

    # preprocess test data
    X_values, y_encoded = preprocessor2.preprocess_test(df, encoder)


    #print(X_values)
    #print(y_encoded)

    mapping = {i: label for i, label in enumerate(encoder.classes_)}
    #print(mapping)



    ##########################################################
    ###################   INFERENCE NEW DATA #################
    ##########################################################


    path = "data/test_data.csv"
    df2 = pd.read_csv(path)
    #df2 = df2.drop('Label', axis=1)


    #print(df.iloc[0].shape) # (79,)

    preprocessor3 = PreprocessorFactory.get_preprocessor(model_name, cfg, cfg.preprocessor)

    # preprocess test data
    X_values_inference = preprocessor3.preprocess_inference(df2)


    #print(X_values_inference)


    ##########################################################
    ###################   PREDICT ONE SAMPLE #################
    ##########################################################


    import numpy as np

    sample = np.random.rand(78)



    preprocessor4 = PreprocessorFactory.get_preprocessor(model_name, cfg, cfg.preprocessor)
    X_simple = preprocessor4.preprocess_single(sample)

    #print(sample)
    #print(X_simple)



    ##########################################################
    ###################   LOAD RANDOM MODEL AND MAKE A PRED ##
    ##########################################################

    run_id = "d4b2a68f989248f6990a9ee78f34d757"
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/tree_model")

    print(model)

    pred = model.predict(X_simple)
    print(pred)






if __name__ == "__main__":
    main()




"""
pred:

tensor([[  159.9129,  -141.3121,  -654.4407,   -77.6352,  -407.2115,  -491.3810,
          -748.4187,  -302.7671,    97.4854, -1398.7358]])


tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 1.0840e-22, 0.0000e+00]])
tensor([0])



"""

