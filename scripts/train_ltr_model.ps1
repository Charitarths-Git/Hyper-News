param(
    [string]$FeaturesCsv = "data/ltr_features.auto.csv",
    [string]$OutputModel = "models/ltr_model.txt",
    [int]$LimitImpressions = 2000
)

$python312 = "C:\Users\rinak\AppData\Local\Programs\Python\Python312\python.exe"
$trainerPython = $(if (Test-Path $python312) { $python312 } else { "python" })

python backend/export_ltr_features.py --source auto --output-csv $FeaturesCsv --limit-impressions $LimitImpressions
& $trainerPython backend/train_ltr.py --features-csv $FeaturesCsv --output-model $OutputModel
