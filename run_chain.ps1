$ErrorActionPreference = "Stop"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONIOENCODING = 'utf-8'
$py = "C:\Users\toont\Documents\GitHub\jlens-tinystories\.venv\Scripts\python.exe"
$src = "C:\Users\toont\Documents\GitHub\jlens-tinystories\src"

$steps = @(
    @("$src\fit_lens.py", "C", "100"),
    @("$src\fit_lens.py", "A", "1000"),
    @("$src\fit_lens.py", "B", "1000"),
    @("$src\fit_lens.py", "C", "1000"),
    @("$src\evaluate.py", "A", "B", "C"),
    @("$src\make_report.py")
)
foreach ($step in $steps) {
    Write-Output ("CHAIN-STEP-START: " + ($step -join " "))
    & $py @step
    if ($LASTEXITCODE -ne 0) {
        # One logged retry per step: fits resume from their checkpoint, so a
        # transient native crash (0xC0000409 cuBLAS race seen on fit B) does
        # not repeat completed work. Hyperparameters unchanged.
        Write-Output ("CHAIN-STEP-RETRY: " + ($step -join " ") + " after exit=$LASTEXITCODE")
        & $py @step
        if ($LASTEXITCODE -ne 0) {
            Write-Output ("CHAIN-STEP-FAILED: " + ($step -join " ") + " exit=$LASTEXITCODE")
            exit 1
        }
    }
    Write-Output ("CHAIN-STEP-DONE: " + ($step -join " "))
}
Write-Output "CHAIN-ALL-DONE"
