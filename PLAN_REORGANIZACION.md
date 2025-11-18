# Plan de Reorganización de Documentación
**Fecha**: 2025-11-17  
**Respaldo**: `backups/pre_reorganizacion_20251117_213012/`

## 📋 Estructura Actual vs Propuesta

### Documentos de Entrega (reports/phase2/)

| Archivo Actual | Tamaño | Acción | Nuevo Nombre |
|----------------|--------|--------|--------------|
| `EXECUTIVO.md` | 11K | ✅ Mover + Renombrar | `entregas/01_resumen_ejecutivo.md` |
| `ieee_paper/paper_final.tex` | 22K | ✅ Mover + Renombrar | `entregas/02_articulo_ieee.tex` |
| `ARTICULO.md` | 23K | ✅ Mover + Renombrar | `entregas/02_articulo_ieee.md` |
| `REPORTE_TECNICO.md` | 55K | ✅ Unificar | `entregas/03_reporte_tecnico.md` |
| `REPORTE_TECNICO_PARTE2.md` | 50K | ✅ Unificar | (contenido integrado en 03) |
| `INFORME_COMPLETO.md` | 45K | ❌ Eliminar | (duplicado, contenido en 03) |
| `RESULTADOS.md` | 45K | ✅ Mover + Renombrar | `resultados_experimentales.md` |
| `BITACORA.md` (raíz) | 17K | ✅ Mover + Renombrar | `entregas/04_bitacora_proyecto.md` |

### Figuras (reports/phase2/figures/)

| Archivo Actual | Acción | Nuevo Nombre |
|----------------|--------|--------------|
| `fig1_class_distribution.*` | ✅ Renombrar | `fig01_distribucion_clases.*` |
| `fig2_length_distribution.*` | ✅ Renombrar | `fig02_longitud_resenas.*` |
| `fig3_f1_comparison.*` | ✅ Renombrar | `fig03_comparacion_f1.*` |
| `fig4_uni_vs_bi.*` | ✅ Renombrar | `fig04_unidireccional_vs_bidireccional.*` |
| `fig5_efficiency.*` | ✅ Renombrar | `fig05_eficiencia.*` |
| `fig6_confusion_matrix.*` | ✅ Renombrar | `fig06_matriz_confusion.*` |
| `fig7_preprocessing_impact.*` | ✅ Renombrar | `fig07_impacto_preprocesamiento.*` |
| `fig8_cudnn_optimization.*` | ✅ Renombrar | `fig08_optimizacion_cudnn.*` |

### Archivos LaTeX (reports/phase2/ieee_paper/)

| Archivo Actual | Acción | Nuevo Destino |
|----------------|--------|---------------|
| `paper_final.tex` | ✅ Mover | `latex/articulo_ieee.tex` |
| `IEEEtai.cls` | ✅ Mover | `latex/IEEEtai.cls` |
| `README.md` | ✅ Actualizar + Mover | `latex/README.md` |
| `paper.tex` | ❌ Eliminar | (versión antigua) |
| `figuras/` (antiguas) | ❌ Eliminar | (obsoletas) |
| `fig1_template.png` | ❌ Eliminar | (obsoleto) |

### Documentación Técnica (docs/phase2/)

| Archivo Actual | Acción | Nuevo Nombre |
|----------------|--------|--------------|
| `PLAN_EXPERIMENTAL.md` | ✅ Renombrar | `01_plan_experimental.md` |
| `COMBINACIONES.md` | ✅ Renombrar | `02_combinaciones.md` |
| `MODELOS.md` | ✅ Renombrar | `03_arquitectura_modelos.md` |
| `PIPELINE.md` | ✅ Renombrar | `04_pipeline.md` |
| `DATASET_RESUMEN.md` | ✅ Renombrar | `05_dataset.md` |

## 🎯 Estructura Final

```
reports/
└── phase2/
    ├── entregas/                          # 📦 Documentos oficiales de entrega
    │   ├── 01_resumen_ejecutivo.md        # Resumen ejecutivo (EXECUTIVO.md)
    │   ├── 02_articulo_ieee.tex           # Artículo IEEE LaTeX (paper_final.tex)
    │   ├── 02_articulo_ieee.md            # Artículo IEEE Markdown (ARTICULO.md)
    │   ├── 03_reporte_tecnico.md          # Reporte técnico unificado
    │   └── 04_bitacora_proyecto.md        # Bitácora del proyecto
    │
    ├── figuras/                           # 📊 Todas las figuras (16 archivos)
    │   ├── fig01_distribucion_clases.pdf
    │   ├── fig01_distribucion_clases.png
    │   ├── fig02_longitud_resenas.pdf
    │   ├── fig02_longitud_resenas.png
    │   ├── fig03_comparacion_f1.pdf
    │   ├── fig03_comparacion_f1.png
    │   ├── fig04_unidireccional_vs_bidireccional.pdf
    │   ├── fig04_unidireccional_vs_bidireccional.png
    │   ├── fig05_eficiencia.pdf
    │   ├── fig05_eficiencia.png
    │   ├── fig06_matriz_confusion.pdf
    │   ├── fig06_matriz_confusion.png
    │   ├── fig07_impacto_preprocesamiento.pdf
    │   ├── fig07_impacto_preprocesamiento.png
    │   ├── fig08_optimizacion_cudnn.pdf
    │   └── fig08_optimizacion_cudnn.png
    │
    ├── latex/                             # 📄 Archivos LaTeX
    │   ├── articulo_ieee.tex
    │   ├── IEEEtai.cls
    │   └── README.md
    │
    └── resultados_experimentales.md       # 📈 Resultados detallados

docs/
└── phase2/
    ├── 01_plan_experimental.md
    ├── 02_combinaciones.md
    ├── 03_arquitectura_modelos.md
    ├── 04_pipeline.md
    └── 05_dataset.md

backups/
└── pre_reorganizacion_20251117_213012/   # 💾 Respaldo completo
    ├── reports/
    ├── docs/
    └── BITACORA.md
```

## 📝 Archivos a Eliminar

- ❌ `reports/phase2/INFORME_COMPLETO.md` (duplicado)
- ❌ `reports/phase2/ieee_paper/paper.tex` (versión antigua)
- ❌ `reports/phase2/ieee_paper/figuras/` (figuras obsoletas)
- ❌ `reports/phase2/ieee_paper/fig1_template.png` (obsoleto)

## 🔄 Cambios en Referencias

Después de la reorganización, actualizar referencias en:

1. **README.md** (raíz) - Actualizar rutas a documentos
2. **latex/README.md** - Actualizar rutas a figuras
3. **scripts/generate_figures.py** - Actualizar ruta de salida (opcional)

## ✅ Checklist de Ejecución

- [x] Crear respaldo en `backups/pre_reorganizacion_*/`
- [ ] Crear estructura de directorios
- [ ] Mover y renombrar archivos de entrega
- [ ] Unificar reporte técnico (PARTE1 + PARTE2)
- [ ] Renombrar figuras (español + numeración)
- [ ] Mover archivos LaTeX
- [ ] Renombrar documentación técnica
- [ ] Eliminar archivos obsoletos
- [ ] Actualizar referencias en README.md
- [ ] Actualizar referencias en latex/README.md
- [ ] Commit de cambios
- [ ] Verificar que todo funciona correctamente

## 📊 Estadísticas

- **Archivos a mover/renombrar**: 29
- **Archivos a eliminar**: 4+
- **Archivos a unificar**: 2 → 1
- **Directorios nuevos**: 2 (entregas/, latex/)
- **Tamaño del respaldo**: 4.9 MB

