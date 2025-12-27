@echo off

chcp 65001
set PYTHONUTF8=1
:: The original source of the webui.bat file is stable-diffusion-webui
:: Modified and enhanced by Gemini with features for venv management and requirements handling.

:: --------- Configuration ---------
set COMMANDLINE_ARGS=

:: Define the application directory (folder name)
:: Leave empty if the app is in the root directory.
set APP_DIR=

:: Define the name of the Launch application
set APPLICATION_NAME=app.py

:: Define the requirements filename, default is requirements.txt
set REQUIREMENTS_FILE=requirements.txt

:: Define the name of the virtual environment directory
set VENV_NAME=venv

:: Set to 1 to always attempt to update packages from requirements.txt on every launch
set ALWAYS_UPDATE_REQS=0
:: ---------------------------------

:: --------- Path Setup Logic ---------
:: Logic to handle paths based on whether APP_DIR is set
if defined APP_DIR (
    set "TARGET_REQ=%~dp0%APP_DIR%\%REQUIREMENTS_FILE%"
    set "TARGET_SCRIPT=%~dp0%APP_DIR%\%APPLICATION_NAME%"
    echo Working in subdirectory: %APP_DIR%
) else (
    set "TARGET_REQ=%~dp0%REQUIREMENTS_FILE%"
    set "TARGET_SCRIPT=%~dp0%APPLICATION_NAME%"
    echo Working in root directory.
)
:: ------------------------------------

:: Set PYTHON executable if not already defined
if not defined PYTHON (set PYTHON=python)
:: Set VENV_DIR using VENV_NAME if not already defined
if not defined VENV_DIR (set "VENV_DIR=%~dp0%VENV_NAME%")

mkdir tmp 2>NUL

:: Check if Python is callable
%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_pip
echo Couldn't launch python
goto :show_stdout_stderr

:check_pip
:: Check if pip is available
%PYTHON% -mpip --help >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :start_venv
:: If pip is not available and PIP_INSTALLER_LOCATION is set, try to install pip
if "%PIP_INSTALLER_LOCATION%" == "" goto :show_stdout_stderr
%PYTHON% "%PIP_INSTALLER_LOCATION%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :start_venv
echo Couldn't install pip
goto :show_stdout_stderr

:start_venv
:: Skip venv creation/activation if VENV_DIR is explicitly set to "-"
if ["%VENV_DIR%"] == ["-"] goto :skip_venv_entirely
:: Skip venv creation/activation if SKIP_VENV is set to "1"
if ["%SKIP_VENV%"] == ["1"] goto :skip_venv_entirely

:: Check if the venv already exists by looking for Python.exe in its Scripts directory
dir "%VENV_DIR%\Scripts\Python.exe" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate_venv_and_maybe_update

:: Venv does not exist, create it
echo Virtual environment not found in "%VENV_DIR%". Creating a new one.
for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv "%VENV_DIR%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% NEQ 0 (
    echo Unable to create venv in directory "%VENV_DIR%"
    goto :show_stdout_stderr
)
echo Venv created.

:: Install requirements for the first time if venv was just created
:: This section handles the initial installation of packages from requirements.txt
:: immediately after a new virtual environment is created.
echo Checking for %REQUIREMENTS_FILE% for initial setup...
if exist "%TARGET_REQ%" (
    echo Found %REQUIREMENTS_FILE% at "%TARGET_REQ%", attempting to install for initial setup...
    call "%VENV_DIR%\Scripts\activate.bat"
    echo Installing packages from %REQUIREMENTS_FILE% ^(initial setup^)...
    "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%TARGET_REQ%"
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install requirements during initial setup. Please check the output above.
        pause
        goto :show_stdout_stderr_custom_pip_initial 
    )
    echo Initial requirements installed successfully.
    call "%VENV_DIR%\Scripts\deactivate.bat"
) else (
    echo No %REQUIREMENTS_FILE% found at "%TARGET_REQ%", skipping package installation.
)
goto :activate_venv_and_maybe_update


:activate_venv_and_maybe_update
:: This label is reached if the venv exists or was just created.
:: Set PYTHON to point to the venv's Python interpreter.
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo Activating venv: %PYTHON%

:: Always update requirements if ALWAYS_UPDATE_REQS is 1
:: This section allows for updating packages from requirements.txt on every launch
:: if the ALWAYS_UPDATE_REQS variable is set to 1.
if defined ALWAYS_UPDATE_REQS (
    if "%ALWAYS_UPDATE_REQS%"=="1" (
        echo ALWAYS_UPDATE_REQS is enabled.
        if exist "%TARGET_REQ%" (
            echo Attempting to update packages from "%TARGET_REQ%"...
            REM No need to call activate.bat here again, PYTHON is already set to the venv's python
            %PYTHON% -m pip install -r "%TARGET_REQ%"
            if %ERRORLEVEL% NEQ 0 (
                echo Failed to update requirements. Please check the output above.
                pause
                goto :endofscript 
            )
            echo Requirements updated successfully.
        ) else (
            echo ALWAYS_UPDATE_REQS is enabled, but no %REQUIREMENTS_FILE% found. Skipping update.
        )
    ) else (
        echo ALWAYS_UPDATE_REQS is not enabled or not set to 1. Skipping routine update.
    )
)

goto :launch

:skip_venv_entirely
:: This label is reached if venv usage is explicitly skipped.
echo Skipping venv.
goto :launch

:launch
:: Launch the main application
echo Launching Web UI with arguments: %COMMANDLINE_ARGS% %*
echo Script path: %TARGET_SCRIPT%
%PYTHON% "%TARGET_SCRIPT%" %COMMANDLINE_ARGS% %*
echo Launch finished.
pause
exit /b

:show_stdout_stderr_custom_pip_initial
:: Custom error handler for failures during the initial pip install process.
echo.
echo exit code ^(pip initial install^): %errorlevel%
echo Errors during initial pip install. See output above.
echo.
echo Launch unsuccessful. Exiting.
pause
exit /b


:show_stdout_stderr
:: General error handler: displays stdout and stderr from the tmp directory.
echo.
echo exit code: %errorlevel%

for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :endofscript
echo.
echo stderr:
type tmp\stderr.txt

:endofscript
echo.
echo Launch unsuccessful. Exiting.
pause
exit /b