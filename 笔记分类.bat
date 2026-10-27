@echo off
chcp 65001 >nul
title 笔记分类器
cd /d "%~dp0"
python client.py %*
pause
