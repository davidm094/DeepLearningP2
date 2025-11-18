# Artículo IEEE - Instrucciones de Compilación

## Archivo Principal
- `articulo_ieee.tex` - Artículo completo en formato IEEE Conference (5-6 páginas)

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
cd reports/phase2/latex
pdflatex articulo_ieee.tex
pdflatex articulo_ieee.tex  # Segunda pasada para referencias
```

El PDF generado será `articulo_ieee.pdf`.

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
2. `fig02_longitud_resenas.pdf` - Distribución de longitud de reseñas por clase
3. `fig03_comparacion_f1.pdf` - Comparación de F1-macro por arquitectura
4. `fig04_unidireccional_vs_bidireccional.pdf` - Comparación uni vs bidireccional
5. `fig05_eficiencia.pdf` - Trade-off F1-macro vs tiempo de entrenamiento
6. `fig06_matriz_confusion.pdf` - Matriz de confusión del mejor modelo
7. `fig07_impacto_preprocesamiento.pdf` - Impacto del preprocesamiento en BiLSTM
8. `fig08_optimizacion_cudnn.pdf` - Impacto de optimización cuDNN

**Nota**: Las rutas en el archivo `.tex` apuntan a `../figuras/figXX_*.pdf`. Si mueves el archivo LaTeX, actualiza estas rutas.

## Tablas Incluidas

- **Tabla I**: Mejor configuración por familia de modelos
- **Tabla II**: Comparación unidireccional vs bidireccional

## Características

- **Formato**: IEEE Conference (dos columnas)
- **Idioma**: Español
- **Extensión**: 5-6 páginas compiladas
- **Palabras clave**: Redes Neuronales Recurrentes, LSTM, GRU, Clasificación de Sentimientos, NLP, Turismo, Español
- **Figuras**: 8 gráficos de alta calidad (300 DPI, PDF vectorial)
- **Tablas**: 2 tablas comparativas
- **Referencias**: 11 citas bibliográficas

## Compilación Online

Si no puedes compilar LaTeX localmente, puedes usar servicios online:

- **Overleaf** (https://www.overleaf.com/)
- **Papeeria** (https://papeeria.com/)

Para usar Overleaf:
1. Crear nuevo proyecto → Upload Project
2. Subir `articulo_ieee.tex`, `IEEEtai.cls`
3. Crear carpeta `figuras/` y subir los 8 PDFs desde `../figuras/`
4. Compilar con pdfLaTeX

## Archivos Necesarios

- `articulo_ieee.tex` - Archivo principal
- `IEEEtai.cls` - Clase IEEE (incluido en este directorio)
- `../figuras/*.pdf` - 8 figuras en formato PDF

## Solución de Problemas

### Error: "File not found: figuras/figXX_*.pdf"
**Solución**: Verifica que las figuras estén en `../figuras/` o actualiza las rutas en el `.tex`.

### Error: "IEEEtai.cls not found"
**Solución**: Asegúrate de que `IEEEtai.cls` esté en el mismo directorio que `articulo_ieee.tex`.

### Advertencias de referencias
**Solución**: Ejecuta `pdflatex` dos veces para resolver referencias cruzadas.

## Versión Markdown

Si prefieres el artículo en formato Markdown, está disponible en:
- `../entregas/02_articulo_ieee.md`

## Contacto

Para más información sobre el proyecto, consulta:
- README principal: `/README.md`
- Reporte técnico completo: `../entregas/03_reporte_tecnico.md`
- Resultados experimentales: `../resultados_experimentales.md`
