# Script de compilacao do Relatorio LaTeX (abnTeX2)
param (
    [switch]$OpenPdf
)
$ErrorActionPreference = "Stop"

$miktexPath = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if (Test-Path $miktexPath) {
    $env:PATH = "$miktexPath;$env:PATH"
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Compilando Relatorio LaTeX (abnTeX2)    " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`n[1/4] Primeira passada: pdflatex..." -ForegroundColor Yellow
& pdflatex -interaction=nonstopmode main.tex | Out-Null

Write-Host "[2/4] Processando referencias: bibtex..." -ForegroundColor Yellow
& bibtex main | Out-Null

Write-Host "[3/4] Segunda passada: pdflatex..." -ForegroundColor Yellow
& pdflatex -interaction=nonstopmode main.tex | Out-Null

Write-Host "[4/4] Terceira passada: pdflatex (resolvendo referencias cruzadas)..." -ForegroundColor Yellow
& pdflatex -interaction=nonstopmode main.tex | Out-Null

$pdf = Get-Item main.pdf
Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "  SUCESSO! PDF gerado com exito:" -ForegroundColor Green
Write-Host "  Arquivo: $($pdf.FullName)" -ForegroundColor White
Write-Host "  Tamanho: $([math]::Round($pdf.Length/1024, 1)) KB" -ForegroundColor White
Write-Host "  Data:    $($pdf.LastWriteTime)" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Green

if ($OpenPdf) {
    Start-Process main.pdf
}
