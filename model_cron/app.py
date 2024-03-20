# import requests

import boto3
import json
import os

# 创建 STS和Sagemaker 客户端
sm_client = boto3.client("sagemaker")


def lambda_handler(event, context):
    """Sample Lambda function reacting to EventBridge events

    Parameters
    ----------
    event: dict, required
        Event Bridge EC2 State Change Events Format

        Event doc: https://docs.aws.amazon.com/eventbridge/latest/userguide/event-types.html#ec2-event-type

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
        The result
    """

    print(event)

    action = event.get('action')

    # 读取环境变量
    model_name = os.environ.get('MODEL_NAME', 'pytorch-inference-llm-v1')
    print(f"model name: {model_name}")

    if action == "start":
        if not start_model(model_name):
            return {
                'statusCode': 200,
                'body': json.dumps({'result': 'Start model fail'})
            }

    elif action == "stop":
        if not stop_model(model_name):
            return {
                'statusCode': 200,
                'body': json.dumps({'result': 'Stop model fail'})
            }
    else:
        return {
            'statusCode': 200,
            'body': json.dumps({'result': 'Action unexpected'})
        }

    return {
        'statusCode': 200,
        'body': json.dumps({'result': 'Success'})
    }


def start_model(model_name):
    region = os.environ.get('AWS_REGION')
    s3_code_artifact = os.environ.get('S3_CODE_ARTIFACT')
    execution_role_arn = os.environ.get('EXECUTION_ROLE_ARN')

    inference_image_uri = (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.24.0-deepspeed0.10.0-cu118"
    )
    if "cn-" in region:
        inference_image_uri = (
            f"727897471807.dkr.ecr.{region}.amazonaws.com.cn/djl-inference:0.24.0-deepspeed0.10.0-cu118"
        )

    print(f"inference image uri: {inference_image_uri}")

    create_model_response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=execution_role_arn,
        PrimaryContainer={
            "Image": inference_image_uri,
            "ModelDataUrl": s3_code_artifact
        },
    )
    model_arn = create_model_response["ModelArn"]

    print(f"Created Model: {model_arn}")

    endpoint_config_name = model_name
    endpoint_name = model_name
    endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "variant1",
                "ModelName": model_name,
                "InstanceType": "ml.g4dn.4xlarge",
                "InitialInstanceCount": 1,
                # "VolumeSizeInGB" : 400,
                # "ModelDataDownloadTimeoutInSeconds": 2400,
                "ContainerStartupHealthCheckTimeoutInSeconds": 15 * 60,
            },
        ],
    )

    print(f"Created endpoint config: {endpoint_config_response['EndpointConfigArn']}")
    print(endpoint_config_response)

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=f"{endpoint_name}", EndpointConfigName=endpoint_config_name
    )
    print(f"Created Endpoint: {create_endpoint_response['EndpointArn']}")

    return True


def stop_model(model_name):
    try:
        # Delete model
        model_response = sm_client.delete_model(ModelName=model_name)
        print(model_response)

        # Delete endpoint config
        config_response = sm_client.delete_endpoint_config(EndpointConfigName=model_name)
        print(config_response)

        # Delete endpoint
        endpoint_response = sm_client.delete_endpoint(EndpointName=model_name)
        print(endpoint_response)

    except Exception as e:
        print(f"Error deleting endpoint '{model_name}': {e}")
        return False

    return True
