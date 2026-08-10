. "$PSScriptRoot/CASEngine.ps1"

Write-Host "Executing Conflict-Aware Synthesis (CAS) Pipeline..." -ForegroundColor Green

$engine = [CASEngine]::new(0.40)
$docs = @(
    [Document]::new("Doc_01", "The FDA approved drug-X for Phase 3 clinical trials in early 2024.", "PeerReviewed_Journal", 0.92, 0.90, 0.95),
    [Document]::new("Doc_02", "Regulators rejected drug-X for Phase 3 trials due to severe health concerns.", "Unverified_Blog", 0.65, 0.75, 0.25),
    [Document]::new("Doc_03", "Phase 3 clinical trial for drug-X was approved by health regulatory bodies.", "Medical_News_Outlet", 0.88, 0.85, 0.80),
    [Document]::new("Doc_04", "Drug-X is an experimental therapeutic compound tested for hypertension.", "Pharma_DB", 0.95, 0.92, 0.90)
)

$result = $engine.ExecutePipeline("Is Drug-X approved?", $docs)

Write-Host "`n" $result.C_Verified -ForegroundColor Cyan
Write-Host ("CAS Framework CPI Score: {0:N2}" -f [CASEngine]::ComputeCPI(82.7, 86.8, 9.5, 6.1)) -ForegroundColor Yellow
