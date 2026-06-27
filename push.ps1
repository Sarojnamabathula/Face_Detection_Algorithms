# Find Git executable
$gitExe = "git"
if (Test-Path "C:\Program Files\Git\cmd\git.exe") {
    $gitExe = "C:\Program Files\Git\cmd\git.exe"
} elseif (Test-Path "C:\Program Files (x86)\Git\cmd\git.exe") {
    $gitExe = "C:\Program Files (x86)\Git\cmd\git.exe"
} elseif (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git is not detected on your system. Please install Git first."
    Exit
}

Write-Host "Using Git from: $gitExe"

# Initialize Git if not already done
if (!(Test-Path .git)) {
    Write-Host "Initializing git repository..."
    & $gitExe init
}

# Configure local author identity if not set globally
$globalEmail = & $gitExe config --global user.email 2>$null
$globalName = & $gitExe config --global user.name 2>$null

if ([string]::IsNullOrEmpty($globalEmail)) {
    Write-Host "Configuring local git user.email..."
    & $gitExe config user.email "sarojnamabathula@users.noreply.github.com"
}
if ([string]::IsNullOrEmpty($globalName)) {
    Write-Host "Configuring local git user.name..."
    & $gitExe config user.name "Sarojnamabathula"
}

# Add remote origin
$remoteUrl = "https://github.com/Sarojnamabathula/Face_Detection_Algorithms.git"
$remoteCheck = & $gitExe remote 2>$null
if ($remoteCheck -contains "origin") {
    Write-Host "Updating remote origin URL..."
    & $gitExe remote set-url origin $remoteUrl
} else {
    Write-Host "Adding remote origin..."
    & $gitExe remote add origin $remoteUrl
}

# Stage files (ignores python-embed automatically via .gitignore)
Write-Host "Staging files..."
& $gitExe add .

# Commit changes
Write-Host "Committing changes..."
& $gitExe commit -m "Add face detection comparison application"

# Rename branch to main
& $gitExe branch -M main

# Push to repository
Write-Host "Pushing to GitHub..."
& $gitExe push -u origin main
