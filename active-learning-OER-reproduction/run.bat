@echo off
rem Run any script with the oer2022clean python.
rem Usage: run.bat audit_data.py
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set TF_CPP_MIN_LOG_LEVEL=2
"D:\Pycharm\Project from Anaconda\oer2022clean\python.exe" %*
