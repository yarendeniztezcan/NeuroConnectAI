Set-Location $PSScriptRoot

Write-Host "NeuroConnectAI AutoSync baslatildi" -ForegroundColor Cyan

while ($true) {
    Write-Host "`nKontrol: $(Get-Date)"

    if ((git branch --show-current) -ne "main") {
        Write-Host "main dalinda degilsiniz."
        break
    }

    git add -A
    if ($LASTEXITCODE -ne 0) { break }

    git diff --cached --quiet

    if ($LASTEXITCODE -eq 1) {
        git commit -m "AutoSync - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        if ($LASTEXITCODE -ne 0) { break }
    }

    git push origin main

    if ($LASTEXITCODE -eq 0) {
        Write-Host "GitHub guncel." -ForegroundColor Green
    } else {
        Write-Host "Push basarisiz. Manuel kontrol gerekli." -ForegroundColor Red
        break
    }

    Start-Sleep -Seconds 300
}