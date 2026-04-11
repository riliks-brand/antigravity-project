<# 
    Elite Trading Bot v3.0 - Auto-Restart Wrapper
    ================================================
    Monitors the bot process and restarts on crash.
    Place this script in the same directory as main.py.
    
    Usage: 
        Right-click -> Run with PowerShell
        OR: powershell -ExecutionPolicy Bypass -File .\run_bot.ps1
#>

$ErrorActionPreference = "Continue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptDir "venv\Scripts\python.exe"
$mainScript = Join-Path $scriptDir "main.py"
$logFile = Join-Path $scriptDir "restart_log.txt"

# Fallback to system python if venv not found
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
    Write-Host "[Wrapper] venv not found. Using system Python." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Elite Trading Bot v3.0 - Auto-Restart" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Python : $pythonExe" -ForegroundColor Gray
Write-Host "  Script : $mainScript" -ForegroundColor Gray
Write-Host "  Log    : $logFile" -ForegroundColor Gray
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

$restartCount = 0
$maxRestarts = 10
$cooldownSeconds = 30

while ($restartCount -lt $maxRestarts) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $restartCount++
    
    Write-Host "[$timestamp] Starting bot (attempt $restartCount/$maxRestarts)..." -ForegroundColor Green
    Add-Content -Path $logFile -Value "[$timestamp] Bot started (attempt $restartCount)"
    
    # Run the bot
    $process = Start-Process -FilePath $pythonExe -ArgumentList $mainScript -WorkingDirectory $scriptDir -NoNewWindow -PassThru -Wait
    
    $exitCode = $process.ExitCode
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    if ($exitCode -eq 0) {
        Write-Host "[$timestamp] Bot exited cleanly (code 0). Stopping wrapper." -ForegroundColor Green
        Add-Content -Path $logFile -Value "[$timestamp] Bot exited cleanly (code 0)"
        break
    }
    
    Write-Host "[$timestamp] Bot CRASHED with exit code $exitCode!" -ForegroundColor Red
    Add-Content -Path $logFile -Value "[$timestamp] Bot crashed (code $exitCode). Restarting in ${cooldownSeconds}s..."
    
    Write-Host "[$timestamp] Restarting in $cooldownSeconds seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds $cooldownSeconds
}

if ($restartCount -ge $maxRestarts) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] MAX RESTARTS ($maxRestarts) REACHED. Manual intervention required." -ForegroundColor Red
    Add-Content -Path $logFile -Value "[$timestamp] MAX RESTARTS reached. Stopping."
}

Write-Host ""
Write-Host "Auto-restart wrapper finished." -ForegroundColor Cyan
