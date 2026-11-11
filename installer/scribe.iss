; Inno Setup script for ScribeSetup.exe. Build with installer\build.ps1, which stages the files first.
; The installer is small: it lays down Scribe and uv, then install.ps1 fetches Python 3.13, PyTorch and the
; speech models on the user's PC. Nothing personal is in it: the stage holds only files tracked in git.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef Stage
  #define Stage "..\dist\stage"
#endif

[Setup]
AppId={{FCDD6DB5-66B5-4F96-BAC8-1F969D2D6E37}
AppName=Scribe
AppVersion={#AppVersion}
AppVerName=Scribe {#AppVersion}
AppPublisher=Scribe contributors
AppPublisherURL=https://github.com/wimlrx/scribe
AppSupportURL=https://github.com/wimlrx/scribe/issues
DefaultDirName={localappdata}\Programs\Scribe
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
LicenseFile={#Stage}\LICENSE
SetupIconFile={#Stage}\scribe.ico
UninstallDisplayIcon={app}\scribe.ico
UninstallDisplayName=Scribe
OutputDir=..\dist
OutputBaseFilename=ScribeSetup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=no

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Run]
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"" -Pause -Start"; \
  Description: "Set up now: downloads Python, PyTorch and the speech models (about 10 GB, 10-30 minutes)"; \
  Flags: postinstall waituntilterminated skipifsilent

[UninstallRun]
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\scribe.pyw"" --quit"; \
  WorkingDir: "{app}"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "QuitScribe"
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"" -Uninstall"; \
  Flags: runhidden waituntilterminated; RunOnceId: "RemoveShortcuts"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\scribe\__pycache__"

[Messages]
FinishedLabel=Scribe's files are installed. If you ticked "Set up now", a window will download what Scribe needs and start it in the system tray. Your recordings and settings stay on this PC (Documents\Scribe and %USERPROFILE%\.scribe) and are kept if you uninstall.

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  if not WizardSilent() and not FileExists(ExpandConstant('{sys}\nvidia-smi.exe')) then
    Result := MsgBox('Scribe needs an NVIDIA graphics card (GeForce RTX 20-series or newer) with a current driver, ' +
      'and none was found on this PC.' + #13#10#13#10 + 'Install anyway?', mbConfirmation, MB_YESNO) = IDYES;
end;
