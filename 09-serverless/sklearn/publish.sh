#!/bin/bash
set -e

REGION="us-west-2"
ECR_URL="162263641347.dkr.ecr.us-west-2.amazonaws.com"
REPO_NAME="churn-prediction-lambda-2"
IMAGE_TAG="v1"

REPO_URL="${ECR_URL}/${REPO_NAME}"
REMOTE_IMAGE_TAG="${REPO_URL}:${IMAGE_TAG}"

aws ecr get-login-password \
  --region ${REGION} \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

echo "Building with classic Docker builder..."

DOCKER_BUILDKIT=0 
docker build \
  --platform linux/amd64 \
  -t ${REMOTE_IMAGE_TAG} \
  .

docker push ${REMOTE_IMAGE_TAG}

echo "Done ✅"