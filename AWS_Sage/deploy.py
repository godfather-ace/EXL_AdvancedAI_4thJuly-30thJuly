import mlflow.sagemaker as mfs

experiment_id = '382535589618420137'
run_id = '33666c25783b475fa7489972d0feec8a'
region = 'us-east-1'
aws_id = '404370322509'
arn = 'arn:aws:iam::404370322509:role/aws-sagemaker-for-deploy-ml-model'
app_name = 'model-application'
model_uri = f'mlruns/{experiment_id}/{run_id}/artifacts/random-forest-model'
tag_id = '2.14.3'


image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:' + tag_id

mfs._deploy(app_name,
	model_uri=model_uri,
	region_name=region,
	mode='create',
	execution_role_arn=arn,
	image_url=image_url)
