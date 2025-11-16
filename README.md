# 📊 DataCluster Analytics Pro

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)

**Plataforma completa de análisis de clustering y minería de datos** con interfaz web interactiva desarrollada con Streamlit y análisis avanzado mediante Jupyter.

## 🎯 Características Principales

### 📈 Algoritmos Implementados

- **PCA** (Análisis de Componentes Principales)
- **HAC** (Clustering Jerárquico Aglomerativo) con validación anti-outliers
- **K-Means** clustering optimizado
- **T-SNE** para visualización no-lineal
- **UMAP** para reducción dimensional avanzada

### 📊 Datasets Incluidos

- **BankChurners.csv** - Análisis de abandono bancario (810 registros)
- **hotel_bookings_muestra.csv** - Análisis de reservas hoteleras (408 registros)

### 🌐 Interfaz Web Interactiva

- Visualizaciones dinámicas con Plotly
- Interpretaciones detalladas de resultados
- Comparaciones algorítmicas en tiempo real
- Navegación intuitiva por secciones

## 🚀 Inicio Rápido

### ⚡ Ejecución con Un Solo Clic

```bash
# Ejecuta el script principal que configura todo automáticamente
launch_project.bat
```

Este script:

1. ✅ Verifica e inicia Docker
2. 🏗️ Construye la imagen con todas las dependencias
3. 🐳 Despliega el contenedor
4. 📦 Valida todas las librerías
5. 🌐 Inicia Streamlit en http://localhost:8501
6. 🔗 Abre automáticamente el navegador

### 📋 Requisitos Previos

- **Docker Desktop** instalado y funcionando
- **Windows 10/11** (los scripts están optimizados para Windows)
- **8GB RAM** recomendados para análisis completos

## 📁 Estructura del Proyecto

```
DataCluster-Analytics-Pro/
├── 📊 streamlit_app.py           # Aplicación web principal
├── 📓 analisis_mineria_datos.ipynb  # Notebook de análisis completo
├── 🧩 paquete_mineria.py         # Módulo de algoritmos personalizados
├── 🚀 launch_project.bat         # Script de inicio unificado
├── 🐳 docker-compose.yml         # Configuración Docker
├── 📦 requirements.txt           # Dependencias Python
├── 📊 BankChurners.csv          # Dataset financiero
├── 🏨 hotel_bookings_muestra.csv # Dataset hotelero
└── 📚 docs/                     # Documentación adicional
```

## 🎮 Guía de Uso

### 1️⃣ Lanzar la Aplicación

```bash
# Método recomendado (todo automático)
launch_project.bat
```

### 2️⃣ Acceder a la Interfaz Web

- 🌐 **URL**: http://localhost:8501
- 🔄 Se abre automáticamente en el navegador

### 3️⃣ Explorar los Análisis

1. **Selección de Dataset** - Elige BankChurners o Hotel Bookings
2. **Análisis Exploratorio** - Revisa estadísticas y correlaciones
3. **PCA** - Reducción dimensional con interpretación
4. **Clustering HAC** - Análisis jerárquico con dendrogramas
5. **K-Means** - Clustering de centroides con métricas
6. **T-SNE/UMAP** - Visualizaciones avanzadas

## 🛠️ Desarrollo y Personalización

### 🔧 Estructura de Código

#### streamlit_app.py

```python
# Aplicación principal con:
- Interfaz de usuario interactiva
- Visualizaciones Plotly integradas
- Interpretaciones automáticas de resultados
- Sistema de caché para optimización
```

#### paquete_mineria.py

```python
# Módulo de algoritmos con:
- Clase AnalisisDatosExploratorio
- Clase NoSupervisado (PCA, HAC, K-Means)
- Validaciones anti-outliers para HAC
- Optimizaciones de rendimiento
```

### 🐳 Configuración Docker

El proyecto usa Docker para garantizar consistencia de dependencias:

```yaml
# docker-compose.yml
- Python 3.9
- Todas las librerías científicas pre-instaladas
- Puerto 8501 para Streamlit
- Volúmenes montados para desarrollo
```

## 📊 Resultados y Análisis

### 🏦 Dataset BankChurners

- **Mejor HAC**: Ward + Euclidean (k=2) - Silhouette: 0.291
- **Mejor K-Means**: k=2 - Silhouette: 0.251
- **Mejor T-SNE**: Perplexity=30, LR=200 - AMI: 0.48
- **Interpretación**: Estructura bimodal con grupo minoritario (7.9%) vs mayoría (92.1%)

### 🏨 Dataset Hotel Bookings

- **Mejor HAC**: Ward + Euclidean (k=4) - Silhouette: 0.782
- **Mejor K-Means**: k=6 - Silhouette: 0.208
- **Mejor T-SNE**: Perplexity=50, LR=200 - AMI: 0.523
- **Interpretación**: HAC superior para segmentación hotelera por patrones complejos

## 🎯 Casos de Uso

### 👥 Para Analistas de Datos

- Análisis exploratorio automatizado
- Comparación de algoritmos de clustering
- Generación de insights de negocio

### 🎓 Para Estudiantes

- Aprendizaje interactivo de algoritmos
- Visualización de conceptos teóricos
- Experimentación con parámetros

### 🏢 Para Empresas

- Segmentación de clientes
- Análisis de patrones de comportamiento
- Reportes ejecutivos automatizados

## 🔧 Comandos Útiles

```bash
# Parar todos los servicios
docker-compose down

# Ver logs del contenedor
docker-compose logs

# Reconstruir la imagen
docker-compose up --build -d

# Acceso directo al contenedor
docker-compose exec mineria-datos bash

# Solo Streamlit (si ya está corriendo Docker)
docker-compose exec mineria-datos streamlit run streamlit_app.py
```

## 🤝 Contribución

### Para Contribuir

1. Fork el repositorio
2. Crea una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añade nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

### 🛡️ Issues y Soporte

- Reporta bugs en la sección Issues
- Solicita nuevas funcionalidades
- Pregunta sobre implementación

## 📜 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👨‍💻 Autores

- **Equipo de Desarrollo** - Análisis inicial y implementación
- **Contribuidores** - Ver [Contributors](https://github.com/username/datacluster-analytics-pro/contributors)

---

## 🌟 Características Destacadas

- ⚡ **Inicio con un clic** - Todo automatizado
- 🎯 **Análisis profesional** - Algoritmos validados científicamente
- 🌐 **Interfaz moderna** - Streamlit con diseño responsivo
- 📊 **Visualizaciones avanzadas** - Plotly interactivo
- 🐳 **Despliegue simple** - Docker containerizado
- 📈 **Interpretaciones detalladas** - Insights automáticos de negocio

---

_Desarrollado con ❤️ para la comunidad de Data Science_
