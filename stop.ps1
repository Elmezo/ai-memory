# ========================================================
# AI Memory Server - Automated Shutdown Script
# ========================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) { $ScriptDir = "D:\ai-memory" }

function Write-Success($msg) {
    Write-Host "  [✓] $msg" -ForegroundColor Green
}

function Write-Info($msg) {
    Write-Host "  [i] $msg" -ForegroundColor Cyan
}

function Write-WarnMsg($msg) {
    Write-Host "  [!] $msg" -ForegroundColor Yellow
}

Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  1. إيقاف خادم الـ REST API (Port 8000)..." -ForegroundColor White
Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray

$foundApi = $false
try {
    # Find processes listening on port 8000
    $connections = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
    if ($connections) {
        $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($p in $pids) {
            try {
                $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                if ($proc) {
                    Write-Info "إيقاف العملية $($proc.ProcessName) (PID: $p)..."
                    Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                    $foundApi = $true
                }
            } catch {}
        }
    }
} catch {}

if ($foundApi) {
    Write-Success "تم إيقاف خادم REST API بنجاح."
} else {
    Write-Info "خادم REST API غير مشغّل حالياً."
}

Write-Host "`n--------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  2. إيقاف حاوية Elasticsearch..." -ForegroundColor White
Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray

try {
    Set-Location $ScriptDir
    Write-Info "جاري إيقاف حاوية Elasticsearch بأمان لحفظ البيانات..."
    docker compose stop 2>&1 | Out-Null
    Write-Success "تم إيقاف حاوية Elasticsearch بنجاح مع الحفاظ على البيانات."
} catch {
    Write-WarnMsg "تعذر الاتصال بـ Docker أو الحاوية متوقفة بالفعل."
}

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "              🛑 تم إيقاف جميع الخدمات بنجاح" -ForegroundColor Cyan
Write-Host "========================================================`n" -ForegroundColor Cyan
