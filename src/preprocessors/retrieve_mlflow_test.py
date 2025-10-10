import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")



def main():
	# get all runs from an experiment
	from mlflow.tracking import MlflowClient

	mlflow.set_tracking_uri("sqlite:///mlflow.db")
	client = MlflowClient()

	experiment = client.get_experiment_by_name("test_prep")
	runs = client.search_runs(experiment.experiment_id)

	for run in runs:
	    print(run.info.run_id, run.data.params)



	#-----------single exp--------------------------

	#run_id = "a656065bf4254e56a87d9102ec02dca9"
	run_id = "1538665d2c6246699e229fdc00d6d7a3"

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




if __name__==main():
	main()