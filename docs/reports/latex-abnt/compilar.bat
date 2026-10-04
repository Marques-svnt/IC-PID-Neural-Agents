@echo off
echo ==========================================
echo   Compilando Relatorio LaTeX (abnTeX2)
echo ==========================================
echo.

set PATH=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64;%PATH%

echo [1/4] Primeira passada: pdflatex...
pdflatex -interaction=nonstopmode main.tex > nul
if errorlevel 1 goto error

echo [2/4] Processando referencias: bibtex...
bibtex main > nul

echo [3/4] Segunda passada: pdflatex...
pdflatex -interaction=nonstopmode main.tex > nul
if errorlevel 1 goto error

echo [4/4] Terceira passada: pdflatex (resolvendo referencias cruzadas)...
pdflatex -interaction=nonstopmode main.tex > nul
if errorlevel 1 goto error

echo.
echo ==========================================
echo   SUCESSO! PDF gerado: main.pdf
echo ==========================================
goto end

:error
echo.
echo [ERRO] Falha na compilacao. Verifique o arquivo main.log.
pause

:end
