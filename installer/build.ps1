# Builds dist\ScribeSetup.exe from the committed tree (git archive HEAD), so nothing untracked or local can
# end up in it. Needs Inno Setup 6 and uv on this PC.
#
#   powershell -ExecutionPolicy Bypass -File installer\build.ps1

$ErrorActionPreference = 'Stop'
$Repo  = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Dist  = Join-Path $Repo 'dist'
$Stage = Join-Path $Dist 'stage'

$Iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Iscc) { throw 'Inno Setup 6 not found (winget install JRSoftware.InnoSetup)' }
$Uv = (Get-Command uv -ErrorAction Stop).Source

if (git -C $Repo status --porcelain) { throw 'commit first: the installer is built from HEAD only' }
$Version = ((Get-Content -LiteralPath (Join-Path $Repo 'pyproject.toml')) -match '^version\s*=' | Select-Object -First 1) -replace '.*"(.*)".*', '$1'
if (-not $Version) { $Version = '0.1.0' }

if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
New-Item -ItemType Directory -Force $Stage | Out-Null
$tar = Join-Path $Dist 'stage.tar'
git -C $Repo archive --format=tar -o $tar HEAD scribe scribe.pyw scribe.ico install.ps1 requirements.txt skill LICENSE README.md installer/THIRD-PARTY-NOTICES.txt
tar -xf $tar -C $Stage
Remove-Item -LiteralPath $tar
Move-Item -LiteralPath (Join-Path $Stage 'installer\THIRD-PARTY-NOTICES.txt') -Destination $Stage
Remove-Item -LiteralPath (Join-Path $Stage 'installer') -Recurse
Copy-Item -LiteralPath $Uv -Destination (Join-Path $Stage 'uv.exe')

& $Iscc /Q "/DAppVersion=$Version" "/DStage=$Stage" (Join-Path $Repo 'installer\scribe.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed ($LASTEXITCODE)" }
$exe = Get-Item -LiteralPath (Join-Path $Dist 'ScribeSetup.exe')
'{0}  {1:N1} MB  sha256 {2}' -f $exe.FullName, ($exe.Length / 1MB), (Get-FileHash -LiteralPath $exe.FullName).Hash
