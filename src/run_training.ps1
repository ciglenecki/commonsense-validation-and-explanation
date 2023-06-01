$lrs = @("1e-5", "1e-6", "1e-7")
$weight_decays = @("0", "1e-5", "1e-3")
$warmup_ratios = @("0.1", "0.25", "0.33")
$augmentation_thresholds = @("0.0", "0.5")

foreach ($lr in $lrs) {
  foreach ($weight_decay in $weight_decays) {
    foreach ($warmup_ratio in $warmup_ratios) {
      foreach ($augmentation_threshold in $augmentation_thresholds) {
        $command = "C:\Users\dev\anaconda3\envs\comVE\python.exe src\train.py --lr $lr --weight_decay $weight_decay --warmup_ratio $warmup_ratio --augmentation_threshold $augmentation_threshold"
        Write-Host $command
        Invoke-Expression $command
      }
    }
  }
}
