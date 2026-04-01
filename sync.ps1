robocopy "d:\Clase-fundamentos-aprendizaje-automatico\WeatherChart" "d:\Clase-fundamentos-aprendizaje-automatico\repo\WeatherChart-Proyecto1-ML" /MIR /XD .git __pycache__ .ipynb_checkpoints /MAX:50000000
if ($LASTEXITCODE -ge 8) {
    Write-Error "Robocopy failed with exit code $LASTEXITCODE"
    exit 1
}
Write-Output "Robocopy completed successfully."
exit 0
