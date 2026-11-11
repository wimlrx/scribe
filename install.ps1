# Scribe installer. Safe to run again; -Uninstall removes this install's shortcuts only.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1                 full install
#   powershell -ExecutionPolicy Bypass -File install.ps1 -SkipModels     skip the model downloads
#   powershell -ExecutionPolicy Bypass -File install.ps1 -NoIntegration  no shortcuts, MCP entry or skill
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall      remove this install's shortcuts
#
# Needs: Windows 10/11 (64-bit), an NVIDIA GPU with a current driver, and Python 3.13 or uv
# (ScribeSetup.exe bundles uv, which fetches Python 3.13 itself).
# Optional: Ollama (dictation clean-up, video notes) and Claude Code (vault hand-off, MCP server).

param([switch]$Uninstall, [switch]$SkipModels, [switch]$NoIntegration, [switch]$Pause, [switch]$Start)

$ErrorActionPreference = 'Stop'

$Root     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv     = Join-Path $Root '.venv'
$Pythonw  = Join-Path $Venv 'Scripts\pythonw.exe'
$Python   = Join-Path $Venv 'Scripts\python.exe'
$Entry    = Join-Path $Root 'scribe.pyw'
$Icon     = Join-Path $Root 'scribe.ico'
$Skill    = Join-Path $Root 'skill\transcribe\SKILL.md'

$Startup  = [Environment]::GetFolderPath('Startup')
$Desktop  = [Environment]::GetFolderPath('Desktop')
$Programs = [Environment]::GetFolderPath('Programs')
$SendTo   = [Environment]::GetFolderPath('SendTo')

function Say($m) { Write-Host $m }
function Done($code) {
    if ($Pause) { Write-Host ''; Read-Host 'Press Enter to close' | Out-Null }
    exit $code
}
trap { Say ''; Say "FAILED   $_"; Done 1 }

# Fixed shortcuts. "Send to" entries for each vault in settings.json are added below.
$Shortcuts = @(
    @{ Name = 'Scribe';                   Dir = $Startup;  Args = '' },
    @{ Name = 'Scribe';                   Dir = $Programs; Args = '' },
    @{ Name = 'Notetaker';                Dir = $Desktop;  Args = '--toggle-notetaker' },
    @{ Name = 'Notetaker';                Dir = $Programs; Args = '--toggle-notetaker' },
    @{ Name = 'Scribe - transcript only'; Dir = $SendTo;   Args = '--vault Local --add' }
)

if ($Uninstall) {
    # Only shortcuts that start this install's scribe.pyw: another Scribe's shortcuts of the same name stay.
    $Shell = New-Object -ComObject WScript.Shell
    $dirs = $Shortcuts | ForEach-Object { $_.Dir } | Select-Object -Unique
    foreach ($d in $dirs) {
        foreach ($f in (Get-ChildItem -LiteralPath $d -Filter '*.lnk' -ErrorAction SilentlyContinue)) {
            $sc = $Shell.CreateShortcut($f.FullName)
            if ($sc.Arguments -like ('"' + $Entry + '"*')) { Remove-Item -LiteralPath $f.FullName; Say "removed  $($f.FullName)" }
        }
    }
    exit 0
}

# ---------------------------------------------------------------- Python environment
$Uv = Join-Path $Root 'uv.exe'
if (-not (Test-Path -LiteralPath $Uv)) { $c = Get-Command uv -ErrorAction SilentlyContinue; $Uv = if ($c) { $c.Source } else { $null } }

if (-not (Test-Path -LiteralPath $Python)) {
    if ($Uv) {
        Say 'python   3.13 (uv fetches it if this PC does not have it)'
        & $Uv venv --python 3.13 --seed --quiet $Venv
    } else {
        $py = Get-Command py -ErrorAction SilentlyContinue
        if ($py) { & py -3.13 -m venv $Venv } else { & python -m venv $Venv }
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Python)) { throw 'Python 3.13 not found. Install it from python.org, then run this again.' }
    Say "venv     $Venv"
}

function Pip([string[]]$a) {
    if ($Uv) { & $Uv pip install --quiet --python $Python @a } else { & $Python -m pip install -q @a }
    if ($LASTEXITCODE -ne 0) { throw "package install failed: $($a -join ' ')" }
}
Say 'packages PyTorch (CUDA 12.8) - about 2.5 GB the first time'
Pip @('torch==2.11.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
Say 'packages Scribe requirements'
Pip @('-r', (Join-Path $Root 'requirements.txt'))
Say 'packages ok'

Push-Location -LiteralPath $Root
try {
    # ------------------------------------------------------------ Models (Scribe never downloads while it runs)
    if (-not $SkipModels) {
        $env:HF_HUB_OFFLINE = '0'
        Say 'models   Whisper large-v3 and large-v3-turbo (about 4.5 GB)'
        & $Python -c "from faster_whisper.utils import download_model as d; [d(m) for m in ('large-v3', 'large-v3-turbo')]"
        if ($LASTEXITCODE -ne 0) { Say 'models   FAILED Whisper download - run this again' }

        $md  = & $Python -c "from scribe import config; print(config.models_dir())"
        $rel = 'https://github.com/k2-fsa/sherpa-onnx/releases/download'
        $emb = Join-Path $md 'wespeaker_en_voxceleb_resnet34_LM.onnx'
        $seg = Join-Path $md 'sherpa-onnx-pyannote-segmentation-3-0\model.onnx'
        New-Item -ItemType Directory -Force $md | Out-Null
        if (-not (Test-Path -LiteralPath $emb)) {
            Say 'models   speaker embedding'
            curl.exe -sSfL -o "$emb.part" "$rel/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
            if ($LASTEXITCODE -eq 0) { Move-Item "$emb.part" $emb -Force }
        }
        if (-not (Test-Path -LiteralPath $seg)) {
            Say 'models   speaker segmentation'
            $tb = Join-Path $env:TEMP 'sherpa-onnx-pyannote-segmentation-3-0.tar.bz2'
            curl.exe -sSfL -o $tb "$rel/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
            if ($LASTEXITCODE -eq 0) { & $Python -I -c "import sys, tarfile; tarfile.open(sys.argv[1]).extractall(sys.argv[2], filter='data')" $tb $md }
            Remove-Item $tb -ErrorAction SilentlyContinue
        }
        foreach ($m in $emb, $seg) { if (Test-Path -LiteralPath $m) { Say "have     $m" } else { Say "FAILED   $m" } }

        # Ollama models are optional: dictation and the Notetaker work without them. The vision model is large,
        # so it is only named here, never pulled unasked.
        if (Get-Command ollama -ErrorAction SilentlyContinue) {
            $names = @(& $Python -c "from scribe import config; s = config.load_settings(); print(s['dictation']['cleanup']['model']); print(s['ollama']['vision_model'])")
            Say "ollama   pull $($names[0])"; ollama pull $names[0]
            Say "ollama   for video notes, run: ollama pull $($names[1])   (large download)"
        } else { Say 'ollama   not found - dictation clean-up and video notes stay off until you install it (ollama.com)' }
    }

    # ------------------------------------------------------------ Icon, inbox folders, settings.json
    if (-not (Test-Path -LiteralPath $Icon)) {
        & $Python -c "from scribe.ui import write_ico; write_ico(r'$Icon')"
        if ($LASTEXITCODE -ne 0) { throw 'icon not written' }
    }
    & $Python -c "from scribe import config; s = config.load_settings(); [print('inbox    ' + str(config.inbox_dir(v, s))) for v in config.vault_names(s)]"
    if ($LASTEXITCODE -ne 0) { throw 'inbox folders not created' }

    # One "Send to" entry per vault in settings.json
    $vaults = & $Python -c "from scribe import config; [print(v) for v in config.load_settings()['vaults']]"
    foreach ($v in $vaults) { if ($v) { $Shortcuts += @{ Name = "Scribe - $v"; Dir = $SendTo; Args = "--vault `"$v`" --add" } } }

    # ------------------------------------------------------------ Claude Code (optional)
    $Claude = Join-Path $env:USERPROFILE '.local\bin\claude.exe'
    if (-not (Test-Path -LiteralPath $Claude)) { $c = Get-Command claude -ErrorAction SilentlyContinue; if ($c) { $Claude = $c.Source } }
    if ($NoIntegration) {
        Say 'claude   skipped (-NoIntegration)'
    } elseif (Test-Path -LiteralPath $Claude) {
        & $Python -m scribe.mcp_server register --claude $Claude
        $dst = Join-Path $env:USERPROFILE '.claude\skills\transcribe'
        if (Test-Path -LiteralPath (Join-Path $dst 'SKILL.md')) {
            Say "skill    kept your existing $dst\SKILL.md (Scribe's is in $Skill)"
        } else {
            New-Item -ItemType Directory -Force $dst | Out-Null
            Copy-Item -LiteralPath $Skill -Destination (Join-Path $dst 'SKILL.md')
            Say "skill    $dst\SKILL.md"
        }
    } else { Say 'claude   Claude Code not found - vault hand-off and the MCP server stay off (claude.com/claude-code)' }
}
finally {
    Pop-Location
}

# ---------------------------------------------------------------- Shortcuts
if ($NoIntegration) {
    Say 'shortcut skipped (-NoIntegration)'
} else {
    $Shell = New-Object -ComObject WScript.Shell
    foreach ($s in $Shortcuts) {
        if (-not (Test-Path -LiteralPath $s.Dir)) { New-Item -ItemType Directory -Path $s.Dir | Out-Null }
        $lnk = Join-Path $s.Dir ($s.Name + '.lnk')
        $sc = $Shell.CreateShortcut($lnk)
        $sc.TargetPath = $Pythonw
        $sc.Arguments = ('"' + $Entry + '" ' + $s.Args).Trim()
        $sc.WorkingDirectory = $Root
        $sc.IconLocation = $Icon + ',0'
        $sc.Description = $s.Name
        $sc.Save()
        Say "shortcut $lnk"
    }
}

Say ''
if ($Start) {
    Start-Process -FilePath $Pythonw -ArgumentList ('"' + $Entry + '"') -WorkingDirectory $Root
    Say 'done. Scribe is running in the system tray; it also starts with Windows.'
} else {
    Say 'done. Start Scribe from the Start menu, or with:'
    Say "  Start-Process `"$Pythonw`" -ArgumentList '`"$Entry`"'"
}
Done 0
