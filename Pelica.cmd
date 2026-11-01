@echo off
rem Pelica chat app: double-click to open. A window appears; the model loads in a few seconds.
start "" "%~dp0target\release\pelica.exe" app %*
