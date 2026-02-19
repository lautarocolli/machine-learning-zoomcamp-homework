FROM public.ecr.aws/lambda/python:3.13

RUN pip install onnxruntime keras-image-helper

COPY lambda_function.py .
COPY clothing_classifier_mobilenet_v2_latest.onnx .

CMD ["lambda_function.lambda_handler"]