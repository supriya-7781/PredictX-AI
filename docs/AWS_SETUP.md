# PredictX — AWS Setup Guide

This guide walks through setting up the AWS components:
1. Amazon S3 bucket
2. AWS IoT Core Thing + certificates
3. IoT Rule (routes MQTT → Lambda)
4. AWS Lambda function

> **You do not need AWS to test the core ML functionality.**
> Set `STORAGE_MODE=local` in `.env` and run the simulator with `--mode mock`.

---

## Prerequisites

- An AWS account
- AWS CLI installed and configured (`aws configure`)
- IAM user or role with the permissions listed below

---

## 1. IAM Policy

Create an IAM policy named `predictx-policy` with these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::YOUR-BUCKET-NAME",
        "arn:aws:s3:::YOUR-BUCKET-NAME/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["iot:Connect", "iot:Publish", "iot:Subscribe", "iot:Receive"],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": "arn:aws:lambda:*:*:function:predictx-handler"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "*"
    }
  ]
}
```

---

## 2. S3 Bucket

```bash
aws s3 mb s3://YOUR-BUCKET-NAME --region us-east-1
```

Update `.env`:
```
S3_BUCKET=YOUR-BUCKET-NAME
STORAGE_MODE=s3
```

---

## 3. AWS IoT Core — Create a Thing

### 3a. Create the Thing

```bash
aws iot create-thing --thing-name predictx-device
```

### 3b. Create certificates

```bash
# From your PredictX project root
mkdir -p certs
aws iot create-keys-and-certificate \
  --set-as-active \
  --certificate-pem-outfile certs/device.pem.crt \
  --public-key-outfile  certs/public.pem.key \
  --private-key-outfile certs/private.pem.key
```

Save the `certificateArn` from the output — you'll need it below.

### 3c. Download the root CA

```bash
curl -o certs/AmazonRootCA1.pem \
  https://www.amazontrust.com/repository/AmazonRootCA1.pem
```

### 3d. Create and attach a policy

Create `iot-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["iot:Connect", "iot:Publish", "iot:Subscribe", "iot:Receive"],
    "Resource": "*"
  }]
}
```

```bash
aws iot create-policy \
  --policy-name predictx-iot-policy \
  --policy-document file://iot-policy.json

aws iot attach-policy \
  --policy-name predictx-iot-policy \
  --target YOUR-CERTIFICATE-ARN

aws iot attach-thing-principal \
  --thing-name predictx-device \
  --principal YOUR-CERTIFICATE-ARN
```

### 3e. Get your IoT endpoint

```bash
aws iot describe-endpoint --endpoint-type iot:Data-ATS
```

Copy the `endpointAddress` and add it to `.env`:
```
IOT_ENDPOINT=abcdefg1234567.iot.us-east-1.amazonaws.com
IOT_CLIENT_ID=predictx-simulator
IOT_CERT=certs/device.pem.crt
IOT_KEY=certs/private.pem.key
IOT_CA=certs/AmazonRootCA1.pem
```

### 3f. Test MQTT connection

```bash
python simulator/sensor_simulator.py --mode mqtt --rows 5
```

---

## 4. IoT Rule → Lambda

### 4a. Create the Lambda function

Package the function:

```bash
# Windows (PowerShell)
cd lambda_function
pip install -r requirements.txt -t package/
Copy-Item handler.py package/
New-Item -ItemType Directory -Force -Path "../models" | Out-Null
Copy-Item -Recurse ../models package/models
Compress-Archive -Path package/* -DestinationPath ../predictx-lambda.zip -Force
cd ..
```

Deploy to Lambda:

```bash
aws lambda create-function \
  --function-name predictx-handler \
  --runtime python3.11 \
  --role arn:aws:iam::YOUR-ACCOUNT-ID:role/YOUR-LAMBDA-ROLE \
  --handler handler.lambda_handler \
  --zip-file fileb://predictx-lambda.zip \
  --timeout 30 \
  --memory-size 256 \
  --environment "Variables={S3_BUCKET=YOUR-BUCKET-NAME,AWS_REGION=us-east-1}"
```

To update after code changes:
```bash
aws lambda update-function-code \
  --function-name predictx-handler \
  --zip-file fileb://predictx-lambda.zip
```

### 4b. Add Lambda permission for IoT

```bash
aws lambda add-permission \
  --function-name predictx-handler \
  --statement-id iot-invoke \
  --action lambda:InvokeFunction \
  --principal iot.amazonaws.com
```

### 4c. Create the IoT Rule

```bash
aws iot create-topic-rule \
  --rule-name predictx_ingest \
  --topic-rule-payload '{
    "sql": "SELECT *, topic() AS _topic, timestamp() AS _iot_ts FROM '\''predictx/sensors/+'\''",
    "description": "Route PredictX sensor data to Lambda",
    "actions": [{
      "lambda": {
        "functionArn": "arn:aws:lambda:us-east-1:YOUR-ACCOUNT-ID:function:predictx-handler"
      }
    }],
    "ruleDisabled": false
  }'
```

---

## 5. Verify end-to-end

1. Start the simulator in MQTT mode:
   ```bash
   python simulator/sensor_simulator.py --mode mqtt --rows 10 --delay 2
   ```

2. Check Lambda logs in CloudWatch:
   ```bash
   aws logs tail /aws/lambda/predictx-handler --follow
   ```

3. Check S3 for results:
   ```bash
   aws s3 ls s3://YOUR-BUCKET-NAME/results/
   ```

4. Open the dashboard and check the **History** tab — results should appear.

---

## Architecture diagram

```
[Sensor Simulator]
       │  MQTT (TLS, port 8883)
       ▼
[AWS IoT Core]
  Topic: predictx/sensors/+
       │  IoT Rule SQL
       ▼
[AWS Lambda: predictx-handler]
  ┌── load Isolation Forest model
  ├── run inference
  └── write JSON result to S3
       │
       ▼
[Amazon S3: YOUR-BUCKET/results/]
       │
       ▼
[Streamlit Dashboard — History tab]
```

---

## Cost estimate (free tier)

| Service | Free tier | Estimated usage |
|---|---|---|
| AWS IoT Core | 500,000 messages/month | ~10,000 msgs in testing |
| Lambda | 1M invocations/month | ~10,000 in testing |
| S3 | 5 GB storage | < 1 MB |
| CloudWatch | 5 GB logs/month | minimal |

All testing for this project should comfortably stay within free tier limits.
