import mlflow
import mlflow.pytorch






mlflow.set_tracking_uri("sqlite:///mlflow.db")



def main():
	# get all runs from an experiment
	from mlflow.tracking import MlflowClient

	mlflow.set_tracking_uri("sqlite:///mlflow.db")
	client = MlflowClient()

	experiment = client.get_experiment_by_name("nn_experiment")
	runs = client.search_runs(experiment.experiment_id)

	for run in runs:
	    print(run.info.run_id, run.data.params)



	#-----------single exp--------------------------
	run_id = "8e47b75a15e6463d8213c01736e90f1a"

	run = client.get_run(run_id)
	print(run.data.params)
	print(run.data.tags)
	print(run.data.metrics)


	local_path = mlflow.artifacts.download_artifacts(run_id=run_id)
	print("Artifacts downloaded to:", local_path)

	local_file = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="class_dist_before_smote.json")
	print(open(local_file).read())



	#
	import joblib
	local_path = mlflow.artifacts.download_artifacts(
	    run_id=run_id, artifact_path="preprocessor/scaler.pkl"
	)
	scaler = joblib.load(local_path)

	print(type(scaler))

	#--------------------MODEL-------------------------
	# Build the artifact URI for the model in the run
	model_uri = f"runs:/{run_id}/model"

	# Load the PyTorch model
	loaded_model = mlflow.pytorch.load_model(model_uri)

	print(type(loaded_model))
	print(loaded_model)








if __name__==main():
	main()