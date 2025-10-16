import torch
import torch.nn as nn
import torch.optim as optim
from src.models.nnet import SimpleNN
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

    #print(X_train)



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
    X_values, y_encoded = preprocessor2.preprocess_test(df, scaler, encoder)


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
    X_values_inference = preprocessor3.preprocess_inference(df2, scaler)


    #print(X_values_inference)


    ##########################################################
    ###################   PREDICT ONE SAMPLE #################
    ##########################################################


    import numpy as np

    sample = np.random.rand(78)



    preprocessor4 = PreprocessorFactory.get_preprocessor(model_name, cfg, cfg.preprocessor)
    X_simple = preprocessor4.preprocess_single(sample, scaler)

    #print(sample)
    #print(X_simple)



    ##########################################################
    ###################   LOAD RANDOM MODEL AND MAKE A PRED ##
    ##########################################################
    run_id = "8e47b75a15e6463d8213c01736e90f1a"
    model_uri = f"runs:/{run_id}/model"

    # Load the PyTorch model
    loaded_model = mlflow.pytorch.load_model(model_uri)

    #print(type(loaded_model))
    #print(loaded_model)

    # convert to tensor
    x_tensor = torch.from_numpy(X_simple).float()


    import torch.nn.functional as F

    with torch.no_grad():
        # raw logits
        pred = loaded_model(x_tensor)
        # Apply softmax to get probabilities
        probs = F.softmax(pred, dim=1)
        print(probs)
        # Get predicted class index
        pred_class = torch.argmax(probs, dim=1)
        print(pred_class)



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

