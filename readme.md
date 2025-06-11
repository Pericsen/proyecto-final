# Sistema Inteligente de Clasificación de Reclamos - San Isidro

**Grupo DataNics** - Proyecto Final 82.02  
*Nabel, Dana (62197) - Peric, Nicolás (59566)*

## Descripción

Sistema automatizado de detección y clasificación de reclamos ciudadanos utilizando procesamiento de lenguaje natural (NLP). El proyecto permite identificar reclamos en redes sociales y clasificarlos por categoría temática para su derivación a las áreas municipales correspondientes, devolviendo los resultados en un archivo .xlsx.

### Características principales:
- **Detector de reclamos**: Identifica si un comentario en redes sociales es un reclamo accionable
- **Clasificador temático**: Clasifica reclamos en 5 categorías: Alumbrado, Higiene Urbana, Obras Públicas, Arbolado Urbano, Infraestructura Pública
- **Pipeline automatizado**: Desde extracción de datos hasta predicción final
- **Modelos basados en BERT**: Utilizando SaBERT-Spanish-Sentiment-Analysis y XGBoost + TFIDF


> La obtención de credenciales de Meta Graph API, [Facebook Developers](https://developers.facebook.com/)

## Estructura del Proyecto

```
tesis-project/
├── data/
│   ├── models/                 # Modelos entrenados
│   │   ├── classifier/         # Modelo clasificador BERT
│   │   └── detector/           # Modelo detector XGBoost
│   ├── processed/              # Datos procesados
│   │   ├── train/              # Datos de entrenamiento
│   │   ├── test/               # Datos de prueba
│   │   └── predict/            # Datos para predicción
│   └── raw/                    # Datos sin procesar
│       ├── instagram/
│       ├── facebook/
│       └── munidigital/
├── notebooks/                  # Jupyter notebooks para análisis
├── scripts/                    # Scripts de ejecución
├── src/                        # Código fuente
│   ├── extract/                # Extracción de datos
│   ├── models/                 # Definición de modelos
│   ├── evaluation/             # Evaluación y métricas
│   └── pipelines/              # Pipelines de procesamiento
└── requirements.txt            # Dependencias
```

## Referencias

- [SaBERT-Spanish-Sentiment-Analysis](https://huggingface.co/VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis)
- [Meta Graph API Documentation](https://developers.facebook.com/docs/graph-api)
- [Documentación completa del proyecto](docs/)
