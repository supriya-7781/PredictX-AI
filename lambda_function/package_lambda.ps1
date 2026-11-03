# package_lambda.ps1
# ===================
# Packages the Lambda function + model files into a deployment ZIP.
# Run from the project root:
#   .\lambda_function\package_lambda.ps1

$ErrorActionPreference = "Stop"

$ROOT = Split-Path $PSScriptRoot -Parent
$PKG_DIR = "$PSScriptRoot\package"
$ZIP_PATH = "$ROOT\predictx-lambda.zip"

Write-Host "[1/4] Cleaning package directory..."
if (Test-Path $PKG_DIR) { Remove-Item -Recurse -Force $PKG_DIR }
New-Item -ItemType Directory -Path $PKG_DIR | Out-Null

Write-Host "[2/4] Installing Lambda dependencies..."
pip install -r "$PSScriptRoot\requirements.txt" -t $PKG_DIR --quiet

Write-Host "[3/4] Copying handler and model files..."
Copy-Item "$PSScriptRoot\handler.py" $PKG_DIR
New-Item -ItemType Directory -Path "$PKG_DIR\models" -Force | Out-Null
Copy-Item "$ROOT\models\isolation_forest.pkl" "$PKG_DIR\models\"
Copy-Item "$ROOT\models\scaler.pkl" "$PKG_DIR\models\"

Write-Host "[4/4] Creating ZIP: $ZIP_PATH"
if (Test-Path $ZIP_PATH) { Remove-Item $ZIP_PATH }
Compress-Archive -Path "$PKG_DIR\*" -DestinationPath $ZIP_PATH

$size = (Get-Item $ZIP_PATH).Length / 1MB
Write-Host "Done! predictx-lambda.zip created ($([math]::Round($size,1)) MB)"
Write-Host "Deploy with:"
Write-Host "  aws lambda update-function-code --function-name predictx-handler --zip-file fileb://predictx-lambda.zip"
