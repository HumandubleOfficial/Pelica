@echo off
rem Starts Pelica Desktop (build it first with: npm run build)
cd /d "%~dp0"
start "" "node_modules\electron\dist\electron.exe" .
