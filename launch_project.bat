@echo off
chcp 65001 >nul
color 0A
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              🎯 DATA MINING ANALYTICS PLATFORM               ║
echo ║                   Análisis Completo de Clustering            ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 🚀 Iniciando entorno completo de análisis...
echo.

REM ======== VERIFICACIÓN DE DOCKER ========
echo [1/6] 🔍 Verificando Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker no está instalado o no está iniciado
    echo.
    echo 💡 Opciones:
    echo   1. Instalar Docker Desktop
    echo   2. Iniciar Docker Desktop si ya está instalado
    echo.
    echo ⏸️ Presiona cualquier tecla cuando Docker esté listo...
    pause
    goto :check_docker_again
)

:check_docker_again
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Desktop no está iniciado
    echo 🚀 Iniciando Docker Desktop automáticamente...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe" 2>nul
    echo.
    echo ⏳ Esperando 20 segundos para que Docker Desktop inicie...
    timeout /t 20 /nobreak >nul
    
    REM Verificar nuevamente
    docker info >nul 2>&1
    if errorlevel 1 (
        echo ❌ Docker Desktop aún no responde
        echo 🔧 Inicia Docker Desktop manualmente y presiona cualquier tecla
        pause
    )
)

echo ✅ Docker está funcionando
echo.

REM ======== CONSTRUCCIÓN Y DESPLIEGUE ========
echo [2/6] 🏗️ Construyendo imagen Docker con todas las dependencias...
docker-compose down --remove-orphans >nul 2>&1
docker-compose up --build -d

if errorlevel 1 (
    echo ❌ Error al construir la imagen Docker
    echo 📋 Revisa los logs con: docker-compose logs
    pause
    exit /b 1
)

echo ✅ Contenedor Docker construido exitosamente
echo.

REM ======== VERIFICACIÓN DE SERVICIOS ========
echo [3/6] 🔍 Verificando que el contenedor está corriendo...
timeout /t 5 /nobreak >nul

docker-compose ps | findstr "Up" >nul
if errorlevel 1 (
    echo ❌ El contenedor no está corriendo
    echo 📋 Estado del contenedor:
    docker-compose ps
    echo.
    echo 📝 Logs del contenedor:
    docker-compose logs --tail=10
    pause
    exit /b 1
)

echo ✅ Contenedor corriendo correctamente
echo.

REM ======== VERIFICACIÓN DE DEPENDENCIAS ========
echo [4/6] 📦 Verificando librerías instaladas...
docker-compose exec -T mineria-datos python -c "
print('📋 Verificando dependencias críticas:')
try:
    import pandas as pd; print('  ✅ pandas:', pd.__version__)
    import numpy as np; print('  ✅ numpy:', np.__version__)
    import sklearn; print('  ✅ scikit-learn:', sklearn.__version__)
    import streamlit as st; print('  ✅ streamlit:', st.__version__)
    import plotly; print('  ✅ plotly:', plotly.__version__)
    try:
        import umap; print('  ✅ umap-learn: Disponible')
    except ImportError:
        print('  ⚠️ umap-learn: No disponible (se instalará en el contenedor)')
    
    print('')
    print('🎯 TODAS LAS DEPENDENCIAS CRÍTICAS ESTÁN LISTAS')
except Exception as e:
    print('❌ Error:', e)
    exit(1)
" 2>nul

if errorlevel 1 (
    echo ❌ Error al verificar dependencias
    echo 🔧 Reconstruyendo imagen con dependencias...
    docker-compose up --build -d
    timeout /t 5 /nobreak >nul
)

echo ✅ Todas las dependencias verificadas
echo.

REM ======== INICIANDO STREAMLIT ========
echo [5/6] 🌐 Iniciando aplicación Streamlit...
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                     🎊 ¡APLICACIÓN LISTA!                   ║
echo ║                                                              ║
echo ║  📊 Streamlit Web App:  http://localhost:8501                ║
echo ║  🐳 Contenedor Docker:  min_estudiocaso1_2-mineria-datos-1   ║
echo ║  📓 Jupyter Notebook:   analisis_mineria_datos.ipynb         ║
echo ║                                                              ║
echo ║  Datasets incluidos:                                         ║
echo ║  • BankChurners.csv (Análisis financiero)                   ║
echo ║  • hotel_bookings_muestra.csv (Análisis hotelero)           ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 🌐 Abriendo navegador automáticamente en 5 segundos...
timeout /t 5 /nobreak >nul

REM Abrir navegador automáticamente
start http://localhost:8501

echo [6/6] 🚀 Ejecutando Streamlit...
echo.
echo 💡 INSTRUCCIONES:
echo   • La aplicación está disponible en: http://localhost:8501
echo   • Usa Ctrl+C para detener Streamlit
echo   • Usa 'docker-compose down' para parar todo el entorno
echo.
echo ═══════════════════════════════════════════════════════════════

REM Ejecutar Streamlit en primer plano con logging mejorado
docker-compose exec mineria-datos streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true

echo.
echo ═══════════════════════════════════════════════════════════════
echo                      👋 STREAMLIT DETENIDO
echo ═══════════════════════════════════════════════════════════════
echo.
echo 🐳 El contenedor Docker sigue corriendo en segundo plano
echo.
echo 📋 Comandos útiles:
echo   • Reiniciar Streamlit:    docker-compose exec mineria-datos streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
echo   • Parar todo el entorno:  docker-compose down
echo   • Ver logs:               docker-compose logs
echo   • Estado del contenedor:  docker-compose ps
echo.
pause