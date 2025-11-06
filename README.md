# 🏦 Financial Data Analysis System

> Sistema completo de análisis financiero con descarga de datos, simulación Monte Carlo y visualizaciones profesionales.

## 📋 Descripción

Sistema modular para obtención y análisis de información bursátil que integra múltiples APIs, procesamiento paralelo, simulaciones de riesgo y generación automática de reportes.

### Características principales

- 📊 **Descarga de datos** desde múltiples fuentes (Alpha Vantage, SimFin)
- ⚡ **Procesamiento paralelo** con mejora de rendimiento del 60%
- 🎲 **Simulación Monte Carlo** configurable para análisis de riesgo
- 📈 **7 tipos de visualizaciones** profesionales
- 📄 **Reportes automáticos** en formato Markdown
- 🔄 **Formato estandarizado** independiente de la fuente de datos
- 🧹 **Limpieza automática** de datos con validación

## 🚀 Instalación

### Requisitos previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar el repositorio**
```bash
git clone <tu-repositorio-url>
cd tarea-bloque1-introduccion
```

2. **Instalar dependencias**
```bash
cd src
pip install -r requirements.txt
```

3. **Configurar API keys**

Crear archivo `.env` en la carpeta `src/` con:
```env
ALPHA_VANTAGE_API_KEY=tu_clave_aqui
SIMFIN_API_KEY=tu_clave_aqui
```

**Obtener API keys gratuitas:**
- Alpha Vantage: https://www.alphavantage.co/support/#api-key
- SimFin: https://simfin.com/

## 💻 Uso

### Modo interactivo (recomendado)

```bash
cd src
python main.py
```

El programa ofrece 6 opciones:
1. Descargar solo precios de acciones
2. Descargar solo índices bursátiles
3. Descargar solo datos fundamentales
4. Descargar todo (secuencial)
5. Descargar todo (paralelo - recomendado)
6. Ejecutar simulación Monte Carlo

### Ejemplo de uso

```python
from portfolio.portfolio import Portfolio

# Crear cartera desde archivos CSV
portfolio = Portfolio.from_csv_files(
    file_paths={
        'AAPL': 'data/AAPL_stock_5y_20240101.csv',
        'MSFT': 'data/MSFT_stock_5y_20240101.csv'
    },
    weights={'AAPL': 0.6, 'MSFT': 0.4}
)

# Generar reporte
report = portfolio.report(include_monte_carlo=True)
print(report)

# Generar visualizaciones
plots = portfolio.plots_report()
```

## 📁 Estructura del Proyecto

```
tarea-bloque1-introduccion/
├── src/
│   ├── downloaders/              # Módulos de descarga de datos
│   │   ├── alpha_vantage_base.py # Clase base para Alpha Vantage
│   │   ├── stock_prices.py       # Descarga de acciones
│   │   ├── index_prices.py       # Descarga de índices
│   │   ├── simfin_downloader.py  # Descarga de fundamentales
│   │   ├── parallel_downloader.py # Descarga paralela
│   │   └── statistics_analyzer.py # Análisis estadístico
│   ├── portfolio/                # Módulos de análisis
│   │   ├── portfolio.py          # DataClass principal
│   │   ├── data_preprocessor.py  # Limpieza de datos
│   │   ├── monte_carlo_runner.py # Simulación interactiva
│   │   ├── visualization_engine.py # Motor de gráficos
│   │   └── flexible_loader.py    # Carga flexible de datos
│   ├── tests/                    # Tests unitarios
│   ├── main.py                   # Punto de entrada
│   └── requirements.txt          # Dependencias
├── ARCHITECTURE_DIAGRAM.md       # Diagrama de arquitectura
└── README.md                     # Este archivo
```

## 🏗️ Arquitectura

### 6 Capas Arquitectónicas

1. **User Interface** - Interfaz interactiva con 6 modos de operación
2. **Orchestration** - Gestión de descargas paralelas y simulaciones
3. **Data Downloaders** - Integración con APIs (rate limiting incluido)
4. **Core Analysis** - Portfolio DataClass con Monte Carlo
5. **Data Processing** - Limpieza, validación y visualización
6. **External Services** - APIs y almacenamiento CSV

**Ver diagrama completo:** [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)

### Formato Estandarizado

Todas las fuentes de datos se convierten automáticamente al formato:
```python
DataFrame(
    index=DatetimeIndex,
    columns=['close', 'open', 'high', 'low', 'volume']
)
```

Esto garantiza compatibilidad independientemente de la API utilizada.

## 📊 Funcionalidades Principales

### 1. Descarga de Datos

- **Acciones**: AAPL, MSFT, GOOGL, etc.
- **Índices**: SPX, NDX, RUT, VIX, DJI
- **Fundamentales**: Income Statement, Balance Sheet, Cash Flow
- **Paralelo**: Hasta 3 descargas simultáneas por tipo

### 2. Análisis de Cartera

- Retornos y volatilidad
- Sharpe ratio
- Maximum drawdown
- Correlación entre activos
- Contribución al riesgo

### 3. Simulación Monte Carlo

**Parámetros configurables:**
- Días a simular (default: 252)
- Número de simulaciones (default: 1000)
- Valor inicial (default: $10,000)
- Tipo: Cartera completa o activos individuales

**Resultados:**
- Distribución de valores finales
- Percentiles (5th, 50th, 95th)
- Probabilidad de pérdida
- Value at Risk (VaR)

### 4. Visualizaciones

1. **Price Evolution** - Evolución normalizada de precios
2. **Returns Distribution** - Histogramas y métricas de riesgo
3. **Correlation Heatmap** - Matriz de correlación
4. **Risk-Return Scatter** - Análisis riesgo-retorno
5. **Portfolio Composition** - Pie chart y barras
6. **Performance Dashboard** - 6 métricas clave
7. **Monte Carlo Results** - Bandas de confianza

### 5. Reportes Automáticos

Genera reportes en Markdown con:
- Composición de cartera
- Análisis de retornos
- Resultados de Monte Carlo
- Advertencias y recomendaciones automáticas

## 🔧 Características Técnicas

### Limpieza de Datos

- Detección automática de columnas de precio y fecha
- Eliminación de outliers (cambios >50% diarios)
- Manejo de valores nulos
- Validación de calidad mínima (>30 datos)

### Rate Limiting

- Alpha Vantage: 12 segundos entre llamadas
- SimFin: Respeta límites de la API
- Reintentos automáticos en caso de error

### Flexibilidad

Acepta múltiples formatos de entrada:
- CSV files
- Excel files
- JSON files
- Pandas DataFrames
- Python dictionaries

## 📈 Ejemplo de Salida

### Reporte generado
```markdown
# Portfolio Analysis Report
**Generated:** 2024-01-15 10:30:00

## Portfolio Overview
**Number of Assets:** 3

| Asset | Weight | Data Points | Date Range |
|-------|--------|-------------|------------|
| AAPL  | 40.0%  | 1258        | 2019-01-01 to 2024-01-15 |
| MSFT  | 35.0%  | 1258        | 2019-01-01 to 2024-01-15 |
| GOOGL | 25.0%  | 1258        | 2019-01-01 to 2024-01-15 |

## Returns Analysis
- **Annualized Return:** 18.45%
- **Annualized Volatility:** 22.31%
- **Sharpe Ratio:** 0.827
```

## 🧪 Tests

```bash
cd src
python -m pytest tests/
```

## 📝 Dependencias

- `requests` - Llamadas a APIs
- `pandas` - Manipulación de datos
- `numpy` - Cálculos numéricos
- `matplotlib` - Visualizaciones
- `seaborn` - Gráficos estadísticos
- `simfin` - API de datos fundamentales
- `python-dotenv` - Gestión de variables de entorno

## 🤝 Contribuciones

Proyecto desarrollado como parte del Bloque 1 - Introducción del programa MIAX.

## 📄 Licencia

Proyecto educativo - MIAX 2024

## 🔗 Enlaces Útiles

- [Alpha Vantage Documentation](https://www.alphavantage.co/documentation/)
- [SimFin Documentation](https://simfin.com/api/v2/documentation/)
- [Diagrama de Arquitectura](ARCHITECTURE_DIAGRAM.md)

---

**Nota**: Este sistema es para fines educativos y de análisis. No constituye asesoramiento financiero.
