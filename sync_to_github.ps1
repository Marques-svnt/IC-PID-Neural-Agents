# sync_to_github.ps1
# ============================================================
# Script de sincronização do projeto IC com GitHub
# Uso: .\sync_to_github.ps1 -Message "descrição do que foi feito"
#
# Requer: token do GitHub com permissão 'repo'
# Após instalar Git com admin, este script pode ser aposentado.
# ============================================================

param(
    [string]$Message = "chore: sync project files",
    [string]$Token   = $env:GITHUB_TOKEN   # Definir variável de ambiente ou passar como parâmetro
)

$owner   = "Marques-svnt"
$repo    = "IC-PID-Neural-Agents"
$base    = $PSScriptRoot   # pasta onde o script está (raiz do IC/)
$apiBase = "https://api.github.com/repos/$owner/$repo/contents"

if (-not $Token) {
    Write-Host "ERRO: Token nao definido. Use -Token ou defina a variavel de ambiente GITHUB_TOKEN." -ForegroundColor Red
    exit 1
}

$headers = @{
    "Authorization"        = "Bearer $Token"
    "Accept"               = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

function Get-RemoteSha {
    param([string]$RepoPath)
    try {
        $r = Invoke-RestMethod -Uri "$apiBase/$RepoPath" -Headers $headers -ErrorAction Stop
        return $r.sha
    } catch { return $null }
}

function Push-File {
    param([string]$LocalPath, [string]$RepoPath)
    $bytes  = [IO.File]::ReadAllBytes($LocalPath)
    $b64    = [Convert]::ToBase64String($bytes)
    $sha    = Get-RemoteSha -RepoPath $RepoPath

    $bodyObj = @{ message = $Message; content = $b64 }
    if ($sha) { $bodyObj.sha = $sha }   # necessário para UPDATE de arquivo existente

    $body = $bodyObj | ConvertTo-Json -Compress
    try {
        $r = Invoke-RestMethod -Uri "$apiBase/$RepoPath" -Method PUT `
             -Headers $headers -Body $body -ContentType "application/json; charset=utf-8" -ErrorAction Stop
        Write-Host "  OK  $RepoPath" -ForegroundColor Green
    } catch {
        Write-Host "  ERR $RepoPath : $_" -ForegroundColor Red
    }
    Start-Sleep -Milliseconds 300
}

# ============================================================
# Arquivos a sincronizar (adicionar novos aqui conforme necessário)
# Formato: @("caminho\local\relativo", "caminho/remoto/no/repo")
# ============================================================
$files = @(
    # Raiz
    @(".gitignore",                     ".gitignore"),

    # Projeto - fases
    @("Projeto\README.md",                                                 "Projeto/README.md"),
    @("Projeto\Fase_1_Revisao_Bibliografica\README.md",                    "Projeto/Fase_1_Revisao_Bibliografica/README.md"),
    @("Projeto\Fase_2_Modelagem_Matematica\README.md",                     "Projeto/Fase_2_Modelagem_Matematica/README.md"),
    @("Projeto\Fase_3_PID_Classico\README.md",                             "Projeto/Fase_3_PID_Classico/README.md"),
    @("Projeto\Fase_4_Redes_Neurais\README.md",                            "Projeto/Fase_4_Redes_Neurais/README.md"),
    @("Projeto\Fase_5_Multi_Agente_LangGraph\README.md",                   "Projeto/Fase_5_Multi_Agente_LangGraph/README.md"),
    @("Projeto\Fase_6_Comparacao_Resultados\README.md",                    "Projeto/Fase_6_Comparacao_Resultados/README.md"),

    # Artigo 01
    @("Artigos_Rascunhos\README.md",                                                         "Artigos_Rascunhos/README.md"),
    @("Artigos_Rascunhos\Artigo_01_PID_Neural_Comparativo\main.tex",                         "Artigos_Rascunhos/Artigo_01_PID_Neural_Comparativo/main.tex"),
    @("Artigos_Rascunhos\Artigo_01_PID_Neural_Comparativo\references.bib",                   "Artigos_Rascunhos/Artigo_01_PID_Neural_Comparativo/references.bib"),
    @("Artigos_Rascunhos\Artigo_01_PID_Neural_Comparativo\sections\01_introduction.tex",     "Artigos_Rascunhos/Artigo_01_PID_Neural_Comparativo/sections/01_introduction.tex"),
    @("Artigos_Rascunhos\Artigo_01_PID_Neural_Comparativo\sections\02_modeling.tex",         "Artigos_Rascunhos/Artigo_01_PID_Neural_Comparativo/sections/02_modeling.tex"),
    @("Artigos_Rascunhos\Artigo_01_PID_Neural_Comparativo\sections\03_pid_classical.tex",    "Artigos_Rascunhos/Artigo_01_PID_Neural_Comparativo/sections/03_pid_classical.tex"),
    @("Artigos_Rascunhos\Artigo_01_PID_Neural_Comparativo\sections\04_neural_pid.tex",       "Artigos_Rascunhos/Artigo_01_PID_Neural_Comparativo/sections/04_neural_pid.tex"),
    @("Artigos_Rascunhos\Artigo_01_PID_Neural_Comparativo\sections\05_multiagent.tex",       "Artigos_Rascunhos/Artigo_01_PID_Neural_Comparativo/sections/05_multiagent.tex"),
    @("Artigos_Rascunhos\Artigo_01_PID_Neural_Comparativo\sections\06_results.tex",          "Artigos_Rascunhos/Artigo_01_PID_Neural_Comparativo/sections/06_results.tex"),
    @("Artigos_Rascunhos\Artigo_01_PID_Neural_Comparativo\sections\07_conclusion.tex",       "Artigos_Rascunhos/Artigo_01_PID_Neural_Comparativo/sections/07_conclusion.tex"),

    # Biblioteca
    @("Biblioteca\README.md",                                              "Biblioteca/README.md"),

    # Exercises
    @("exercises\heat_exchanger.py",                  "exercises/heat_exchanger.py"),
    @("exercises\lab02_spyder_interativo.py",          "exercises/lab02_spyder_interativo.py"),
    @("exercises\relatorio_laboratorio_02.md",         "exercises/relatorio_laboratorio_02.md"),
    @("exercises\requirements.txt",                    "exercises/requirements.txt")
)

# ============================================================
# Detectar automaticamente arquivos Python novos em Projeto/
# ============================================================
Get-ChildItem -Path "$base\Projeto" -Recurse -Include "*.py","*.ipynb" | ForEach-Object {
    $rel = $_.FullName.Replace("$base\", "").Replace("\", "/")
    $loc = $_.FullName.Replace("$base\", "")
    if ($files -notcontains @($loc, $rel)) {
        $files += ,@($loc, $rel)
    }
}

# ============================================================
# Executar upload
# ============================================================
Write-Host "`n[sync_to_github] Iniciando sync: $Message" -ForegroundColor Cyan
Write-Host "Repo: https://github.com/$owner/$repo`n"

$ok = 0; $err = 0
foreach ($f in $files) {
    $full = Join-Path $base $f[0]
    if (Test-Path $full) {
        Push-File -LocalPath $full -RepoPath $f[1]
        $ok++
    } else {
        Write-Host "  SKIP (nao existe): $($f[0])" -ForegroundColor DarkGray
    }
}

Write-Host "`n[sync_to_github] Concluido: $ok arquivos sincronizados." -ForegroundColor Cyan
Write-Host "Ver em: https://github.com/$owner/$repo"
