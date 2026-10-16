@echo off
REM One-command demo for Windows. Run from the repo root.
if exist .venv\Scripts\python.exe (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% run_demo.py %*
