# ========================================================
# AI Memory Server - Automated Startup & Health Check Script
# ========================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) { $ScriptDir = "D:\ai-memory" }
$ApiDir = Join-Path $ScriptDir "memory-api"
$VenvPython = Join-Path $ApiDir ".venv\Scripts\python.exe"

function Write-Success($msg) {
    Write-Host "  [✓] $msg" -ForegroundColor Green
}

function Write-WarningMsg($msg) {
    Write-Host "  [!] $msg" -ForegroundColor Yellow
}

function Write-Fail($msg) {
    Write-Host "  [✗] $msg" -ForegroundColor Red
}

function Write-Info($msg) {
    Write-Host "  [i] $msg" -ForegroundColor Cyan
}

Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  1. فحص وتشغيل Docker Desktop..." -ForegroundColor White
Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray

$dockerOk = $false
try {
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $dockerOk = $true }
} catch {}

if (-not $dockerOk) {
    Write-WarningMsg "Docker Desktop غير مشغّل، جاري تشغيله الآن..."
    $dockerExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerExe) {
        Start-Process $dockerExe
        Write-Info "تم إطلاق Docker Desktop، في انتظار استجابة المحرك..."
        $maxWait = 60
        while ($maxWait -gt 0 -and -not $dockerOk) {
            Start-Sleep -Seconds 3
            $maxWait -= 3
            try {
                docker info 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) { $dockerOk = $true }
            } catch {}
        }
    } else {
        Write-Fail "لم يتم العثور على Docker Desktop في المسار القياسي!"
    }
}

if ($dockerOk) {
    Write-Success "Docker Desktop يعمل وجاهز."
} else {
    Write-Fail "فشل الاتصال بـ Docker Daemon. يرجى فتح Docker Desktop يدوياً."
}

Write-Host "`n--------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  2. فحص وتشغيل حاوية Elasticsearch (Port 9200)..." -ForegroundColor White
Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray

$esReady = $false
if ($dockerOk) {
    Set-Location $ScriptDir
    docker compose up -d 2>&1 | Out-Null
    
    $retries = 20
    Write-Info "التحقق من اتصال Elasticsearch..."
    while ($retries -gt 0 -and -not $esReady) {
        try {
            $resp = Invoke-RestMethod -Uri "http://localhost:9200" -TimeoutSec 3 -ErrorAction Stop
            if ($resp.cluster_name) {
                $esReady = $true
            }
        } catch {
            Start-Sleep -Seconds 2
            $retries--
        }
    }

    if ($esReady) {
        $docCount = 0
        try {
            $countResp = Invoke-RestMethod -Uri "http://localhost:9200/agent_memory/_count" -ErrorAction Stop
            if ($countResp -and $countResp.count) {
                $docCount = $countResp.count
            }
        } catch {}
        Write-Success "Elasticsearch 9.2.0 يعمل بنجاح (عدد الذكريات المخزنة: $docCount)."
    } else {
        Write-Fail "تعذر الاتصال بـ Elasticsearch على http://localhost:9200"
    }
} else {
    Write-Fail "تم تخطي تشغيل Elasticsearch لعدم توفر Docker."
}

Write-Host "`n--------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  3. فحص وتشغيل خدمة Ollama ونموذج bge-m3..." -ForegroundColor White
Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray

$ollamaReady = $false
try {
    $tagsResp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
    $ollamaReady = $true
} catch {
    Write-WarningMsg "Ollama غير مشغّل، جاري تشغيله..."
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
    $retries = 10
    while ($retries -gt 0 -and -not $ollamaReady) {
        Start-Sleep -Seconds 2
        try {
            $tagsResp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
            $ollamaReady = $true
        } catch {
            $retries--
        }
    }
}

if ($ollamaReady) {
    $models = $tagsResp.models.name
    $hasBge = $models -match "bge-m3"
    if ($hasBge) {
        Write-Success "Ollama يعمل وموديل bge-m3 متوفر (1024 dims دلالي متعدد اللغات)."
    } else {
        Write-WarningMsg "Ollama يعمل لكن لم يتم العثور على bge-m3! جاري تحميله (ollama pull bge-m3)..."
        ollama pull bge-m3
    }
} else {
    Write-Fail "تعذر تشغيل خدمة Ollama على http://localhost:11434"
}

Write-Host "`n--------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  4. فحص وتشغيل Memory REST API (Port 8000)..." -ForegroundColor White
Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray

$apiReady = $false
try {
    $healthResp = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop
    if ($healthResp.status -eq "ok") {
        $apiReady = $true
    }
} catch {
    Write-Info "الـ REST API غير مشغّل، جاري تشغيله في نافذة منفصلة..."
    $startScript = Join-Path $ApiDir "start_api.bat"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$startScript`"" -WindowStyle Minimized
    $retries = 15
    while ($retries -gt 0 -and -not $apiReady) {
        Start-Sleep -Seconds 2
        try {
            $healthResp = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop
            if ($healthResp.status -eq "ok") {
                $apiReady = $true
            }
        } catch {
            $retries--
        }
    }
}

if ($apiReady) {
    Write-Success "REST API يعمل: http://localhost:8000/docs"
} else {
    Write-Fail "تعذر تشغيل REST API على http://localhost:8000"
}

Write-Host "`n--------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  5. فحص تكامل خادم MCP لـ OpenCode..." -ForegroundColor White
Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray

$mcpCheck = $false
if (Test-Path $VenvPython) {
    Push-Location $ApiDir
    $mcpOut = & $VenvPython -c "import os; os.environ.update({'ES_URL':'http://localhost:9200','EMBEDDING_PROVIDER':'ollama','EMBEDDING_MODEL':'bge-m3','EMBEDDING_DIMS':'1024','OLLAMA_BASE_URL':'http://localhost:11434','MEMORY_TENANT_ID':'opencode_tenant','MEMORY_USER_ID':'opencode_user','MEMORY_AGENT_ID':'opencode','MEMORY_PROJECT_ID':'default','AUTH_MODE':'disabled'}); from mcp_server import memory_health; print('MCP_OK' if memory_health().get('status')=='ok' else 'MCP_FAIL')" 2>&1
    Pop-Location
    if ($mcpOut -match "MCP_OK") {
        $mcpCheck = $true
    }
}

if ($mcpCheck) {
    Write-Success "خادم MCP وأدواته السبعة جاهزة بالكامل لـ OpenCode (stdio)."
} else {
    Write-WarningMsg "تحذير في تهيئة أدوات MCP: $mcpOut"
}

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "              🎉 النتيجة النهائية للنظام" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

if ($dockerOk -and $esReady -and $ollamaReady -and $apiReady -and $mcpCheck) {
    Write-Host "`n  🌟 كل الخدمات تعمل بنجاح 100%! يمكنك الآن فتح OpenCode والعمل فوراً." -ForegroundColor Green
    Write-Host "`n  • رابط التوثيق التفاعلي (Swagger UI): http://localhost:8000/docs" -ForegroundColor Yellow
    Write-Host "  • خادم OpenCode MCP: متصل تلقائياً وجاهز للاستدعاء الدلالي." -ForegroundColor Yellow
} else {
    Write-Host "`n  ⚠️ هناك بعض الخدمات التي تحتاج مراجعة." -ForegroundColor Yellow
}
Write-Host "`n========================================================`n" -ForegroundColor Cyan
