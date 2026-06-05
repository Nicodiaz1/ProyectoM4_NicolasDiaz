# Predicción de Churn Bancario — Modelos Supervisados y No Supervisados

Proyecto de machine learning end-to-end para predecir el **abandono de clientes
(churn)** en un banco y, en paralelo, segmentar la cartera para diseñar estrategias
de retención.

Trabajé sobre el dataset `Churn_Modelling.csv` (clientes de un banco con datos
demográficos, de producto y de actividad) y recorrí el problema desde un modelo
base simple hasta un ensamble optimizado, sumando un análisis no supervisado para
entender el comportamiento de los clientes más allá de la etiqueta de churn.

## Problema que resuelve

Retener un cliente es mucho más barato que conseguir uno nuevo. Si el banco puede
anticipar **qué clientes están por irse**, puede actuar antes con ofertas o
contacto proactivo. El proyecto ataca dos preguntas:

1. *¿Podemos predecir si un cliente va a abandonar?* → clasificación supervisada.
2. *¿Qué perfiles de cliente existen y cómo se relacionan con el churn?* →
   clustering no supervisado.

## Enfoque

El trabajo está dividido en tres notebooks, cada uno una etapa del problema:

### 1. `1_EDA_RegresionLogistica.ipynb` — EDA + modelo base
- Análisis exploratorio de las variables y su relación con la variable objetivo.
- Preprocesamiento con `StandardScaler` (ajustado solo en train para evitar *data leakage*).
- **Regresión Logística** como baseline interpretable.
- Evaluación con accuracy, precision, recall, F1, matriz de confusión y curva ROC / AUC.

### 2. `2_GradientBoosting_Optimizacion.ipynb` — Modelos avanzados
- **Random Forest** como referencia con validación cruzada.
- Optimización de **XGBoost**, **LightGBM** y **CatBoost** con `GridSearchCV` y early stopping.
- **Stacking** de los tres modelos de boosting con una Regresión Logística como meta-modelo.
- Comparación final en test con curvas ROC y Precision-Recall.
- Análisis de **importancia de variables** y validación con `StratifiedKFold` (k=5)
  para confirmar la estabilidad del mejor modelo.

> El **Stacking** obtuvo el mejor ROC-AUC en test: al combinar tres modelos fuertes,
> el meta-modelo aprende qué predicciones son más confiables y reduce sesgo y varianza.

### 3. `3_AprendizajeNoSupervisado.ipynb` — Segmentación de clientes
- **K-Means** con selección de K por método del codo y coeficiente de silueta.
- **DBSCAN** (clustering por densidad) para detectar grupos de forma arbitraria y outliers.
- **PCA** y **t-SNE** para reducir dimensionalidad y visualizar los segmentos en 2D.
- Lectura de los clusters contra el churn real para obtener insights accionables.

## Estructura del repositorio

```
ProyectoM4_NicolasDiaz/
├── 1.Notebooks/
│   ├── 1_EDA_RegresionLogistica.ipynb
│   ├── 2_GradientBoosting_Optimizacion.ipynb
│   └── 3_AprendizajeNoSupervisado.ipynb
└── 2.Documentacion/
    └── Reporte_Modelos.pdf        # Informe con resultados y conclusiones
```

## Tecnologías

- **Python**, **pandas**, **NumPy**
- **scikit-learn** — Logistic Regression, Random Forest, K-Means, DBSCAN, PCA, t-SNE, GridSearchCV
- **XGBoost**, **LightGBM**, **CatBoost** — modelos de gradient boosting
- **Matplotlib** y **Seaborn** — visualización
- **Jupyter Notebook**

## Cómo usarlo

1. Clonar el repo e instalar dependencias:
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn
   ```
2. Ubicar el dataset `Churn_Modelling.csv` en la carpeta del proyecto (ajustar la ruta
   de lectura al inicio de cada notebook).
3. Ejecutar los notebooks en orden (1 → 2 → 3).
