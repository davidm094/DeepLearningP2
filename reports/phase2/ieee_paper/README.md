# Artículo IEEE - Instrucciones de Compilación

## Archivo Principal
- `paper_final.tex` - Artículo completo en formato IEEE (4-6 páginas)

## Requisitos para Compilación

Para compilar el artículo a PDF, necesitas tener instalado LaTeX:

```bash
# En Ubuntu/Debian
sudo apt-get install texlive-full

# O versión mínima
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-lang-spanish
```

## Compilación

```bash
cd reports/phase2/ieee_paper
pdflatex paper_final.tex
pdflatex paper_final.tex  # Segunda pasada para referencias
```

## Contenido del Artículo

El artículo incluye:

1. **Abstract** (español) - Resumen del estudio y contribuciones principales
2. **Introducción** - Motivación, casos de uso, y contribuciones
3. **Trabajo Relacionado** - Literatura sobre RNNs, bidireccionalidad, y análisis de sentimientos en español
4. **Metodología** - Dataset, diseño experimental (DoE), arquitectura de modelos, entrenamiento
5. **Resultados** - Resumen global, análisis de bidireccionalidad, mejor modelo (BiLSTM C02), impacto de preprocesamiento, análisis de eficiencia
6. **Discusión** - Importancia de bidireccionalidad, preprocesamiento mínimo, comparación LSTM vs GRU, limitaciones
7. **Conclusiones** - Contribuciones, recomendaciones, impacto esperado, trabajo futuro
8. **Referencias** - 11 referencias bibliográficas

## Figuras Incluidas

El artículo referencia 8 figuras en formato PDF (ubicadas en `../figuras/`):

1. `fig01_distribucion_clases.pdf` - Distribución de clases en el dataset
2. `fig2_length_distribution.pdf` - Distribución de longitud de reseñas por clase
3. `fig3_f1_comparison.pdf` - Comparación de F1-macro por arquitectura
4. `fig4_uni_vs_bi.pdf` - Comparación unidireccional vs bidireccional
5. `fig5_efficiency.pdf` - Trade-off F1-macro vs tiempo de entrenamiento
6. `fig6_confusion_matrix.pdf` - Matriz de confusión del mejor modelo
7. `fig7_preprocessing_impact.pdf` - Impacto del preprocesamiento en BiLSTM
8. `fig8_cudnn_optimization.pdf` - Impacto de optimización cuDNN

## Tablas Incluidas

- **Tabla I**: Mejor configuración por familia de modelos
- **Tabla II**: Comparación unidireccional vs bidireccional

## Extensión

El artículo compilado tendrá aproximadamente 5-6 páginas en formato IEEE de dos columnas, cumpliendo con el requisito de 4-6 páginas.

## Características

- **Formato**: IEEE Conference (dos columnas)
- **Idioma**: Español
- **Palabras clave**: Redes Neuronales Recurrentes, LSTM, GRU, Clasificación de Sentimientos, NLP, Turismo, Español
- **Figuras**: 8 gráficos de alta calidad (300 DPI)
- **Tablas**: 2 tablas comparativas
- **Referencias**: 11 citas bibliográficas

## Nota

Si no puedes compilar LaTeX localmente, puedes usar servicios online como:
- Overleaf (https://www.overleaf.com/)
- Papeeria (https://papeeria.com/)

Simplemente sube `paper_final.tex`, `IEEEtai.cls` y las figuras PDF.

