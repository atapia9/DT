# Telecom X Parte 2: Predicción de Cancelación de Clientes

https://colab.research.google.com/drive/1-hg88slPvfrJr0UtNdhRkwyvSX0uYDJ-?usp=sharing

## Descripción del Proyecto

Este proyecto corresponde a la segunda parte del Challenge Telecom X, desarrollado por Jesus Armando Tapia Gallegos. Se enfoca en la construcción de modelos predictivos para anticipar la cancelación de clientes (churn) en Telecom X, utilizando datos previamente tratados en la Parte 1 del proyecto.

El objetivo principal es desarrollar y evaluar modelos de clasificación que permitan identificar clientes con alto riesgo de cancelación, apoyando decisiones estratégicas de retención. El análisis se basa en el conjunto de datos tratado, garantizando consistencia y trazabilidad a lo largo del proyecto.

**Autor:** Jesus Armando Tapia Gallegos  
**Fecha:** 11 de febrero, 2026  
**Enfoque:** Predicción de churn utilizando machine learning supervisado.

## Características

- **Carga y Comprensión de Datos:** Importación del dataset tratado desde la Parte 1, con verificación de estructura y calidad.
- **Análisis Exploratorio de Datos (EDA):** Visualización de distribuciones, correlaciones y análisis de desbalance de clases.
- **Preparación de Datos:** Codificación de variables categóricas (One-Hot Encoding), escalado de variables numéricas y división en conjuntos de entrenamiento y prueba.
- **Modelado Predictivo:** Implementación de modelos como Regresión Logística y Árbol de Decisión.
- **Evaluación de Modelos:** Uso de métricas como accuracy, recall, AUC-ROC y matrices de confusión, con énfasis en el recall para identificar churn.
- **Análisis de Importancia de Variables:** Identificación de factores clave que influyen en la cancelación, como antigüedad del cliente, cargos mensuales y tipo de contrato.
- **Interpretación y Recomendaciones:** Conclusiones basadas en los resultados para estrategias de retención.

## Instalación

### Requisitos del Sistema
- Python 3.7 o superior
- Jupyter Notebook o JupyterLab para ejecutar el notebook

### Dependencias
Las siguientes bibliotecas de Python son necesarias:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Instrucciones de Instalación
1. Clona o descarga el repositorio.
2. Instala las dependencias utilizando pip:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Abre el notebook `notebooks/DT_JATG_telecomx_P2.ipynb` en Jupyter Notebook.

Nota: El dataset tratado (`datos_tratados.csv`) se carga automáticamente desde una URL en el notebook. Asegúrate de tener conexión a internet para la carga inicial.

## Uso

1. Ejecuta las celdas del notebook en orden secuencial.
2. El notebook incluye secciones para:
   - Carga de datos
   - Análisis exploratorio
   - Preparación de datos
   - Entrenamiento y evaluación de modelos
   - Análisis de resultados
3. Los resultados se guardan en la carpeta `outputs/`, incluyendo figuras y tablas generadas.

Ejemplo de ejecución básica:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Cargar datos
df = pd.read_csv('data/datos_tratados.csv')

# Preparar datos (simplificado)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Predecir
predictions = model.predict(X_test)
```

## Contribuciones

¡Las contribuciones son bienvenidas! Para contribuir:

1. Haz un fork del repositorio.
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y commitea (`git commit -am 'Agrega nueva funcionalidad'`).
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`).
5. Abre un Pull Request.

Por favor, asegúrate de que tu código siga las mejores prácticas de Python y esté bien documentado.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## Estructura del Proyecto

```
.
├── data/
│   └── datos_tratados.csv          # Dataset tratado
├── data_processed/                 # Datos procesados (si aplica)
├── data_raw/                       # Datos crudos (si aplica)
├── notebooks/
│   └── DT_JATG_telecomx_P2.ipynb   # Notebook principal
├── outputs/
│   ├── figures/                    # Figuras generadas
│   └── tables/                     # Tablas generadas
└── README.md                       # Este archivo
```

## Contacto

Para preguntas o sugerencias, contacta a Jesus Armando Tapia Gallegos.
