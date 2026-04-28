param(
    [string]$ImagePath = ""
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..")

if ([string]::IsNullOrWhiteSpace($ImagePath)) {
    python main.py --output-dir outputs
} else {
    python main.py --image $ImagePath --output-dir outputs
}
