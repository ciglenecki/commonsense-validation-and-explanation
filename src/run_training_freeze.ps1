$augmentation_thresholds = @("0.0", "0.5")
$augmenters = @("none", "rand")
$additional_parameters_list = @("--freeze-bert", "--lr 1e-6", "--lr 5e-5", "--freeze-bert --lr 1e-4", "--freeze-bert --lr 1e-5", "--freeze-bert --lr 5e-5",
"--epochs 3", "--freeze-bert --epochs 3", "--lr 1e-6 --epochs 3", "--lr 5e-5 --epochs 3", "--freeze-bert --lr 1e-4 --epochs 3", "--freeze-bert --lr 1e-5 --epochs 3",
"--freeze-bert --lr 5e-5 --epochs 3")

foreach ($augmentation_threshold in $augmentation_thresholds) {
  foreach ($augmenter in $augmenters) {
    if (($augmentation_threshold -eq "0.0") -and !($augmenter -eq "none")) {continue}
    if (!($augmentation_threshold -eq "0.0") -and ($augmenter -eq "none")) {continue}
    foreach ($additional_parameters in $additional_parameters_list)
    {
      $command = "C:\Users\dev\anaconda3\envs\comVE\python.exe .\src\train.py $additional_parameters --augmenter $augmenter --augmentation_threshold $augmentation_threshold"
      Write-Host $command
      Invoke-Expression $command
    }
  }
}
