#!/usr/bin/env python3
"""
Script para crear un diagrama de arquitectura del modelo RNN.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Crear figura
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Colores
color_input = '#E8F4F8'
color_embedding = '#B3E5FC'
color_rnn = '#4FC3F7'
color_dropout = '#FFE082'
color_dense = '#81C784'
color_output = '#C8E6C9'

# Función para crear cajas
def create_box(ax, x, y, width, height, text, color, fontsize=11, fontweight='normal'):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, wrap=True)

# Función para crear flechas
def create_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax.add_patch(arrow)

# Título
ax.text(5, 13.5, 'Arquitectura de Modelos RNN', ha='center', va='center',
        fontsize=16, fontweight='bold')

# Capa 1: Input
create_box(ax, 3, 12, 4, 0.8, 'Input\nSecuencias tokenizadas\n(batch_size, max_len)', color_input, fontsize=10)
create_arrow(ax, 5, 12, 5, 11.2)

# Capa 2: Embedding
create_box(ax, 2.5, 10.2, 5, 1, 'Embedding Layer\nDimensión: 128 (learned) o 256 (Word2Vec)\nVocabulario: 30,000 palabras', color_embedding, fontsize=10)
create_arrow(ax, 5, 10.2, 5, 9.2)

# Bifurcación: Unidireccional vs Bidireccional
ax.text(5, 8.8, 'Arquitectura RNN', ha='center', va='center',
        fontsize=12, fontweight='bold')

# Rama izquierda: Unidireccional
ax.text(2.5, 8.3, 'Unidireccional', ha='center', va='center',
        fontsize=11, fontweight='bold', style='italic')
create_box(ax, 1, 7.2, 3, 1.2, 'SimpleRNN\n128 unidades\ndropout=0.2', color_rnn, fontsize=9)
create_box(ax, 1, 5.8, 3, 1.2, 'LSTM\n64 unidades\ndropout=0.0', color_rnn, fontsize=9)
create_box(ax, 1, 4.4, 3, 1.2, 'GRU\n64 unidades\ndropout=0.0', color_rnn, fontsize=9)

# Rama derecha: Bidireccional
ax.text(7.5, 8.3, 'Bidireccional', ha='center', va='center',
        fontsize=11, fontweight='bold', style='italic')
create_box(ax, 6, 7.2, 3, 1.2, 'BiSimpleRNN\n128 u. × 2 dir.\ndropout=0.2', color_rnn, fontsize=9)
create_box(ax, 6, 5.8, 3, 1.2, 'BiLSTM\n64 u. × 2 dir.\ndropout=0.0', color_rnn, fontsize=9)
create_box(ax, 6, 4.4, 3, 1.2, 'BiGRU\n64 u. × 2 dir.\ndropout=0.0', color_rnn, fontsize=9)

# Flechas convergentes
create_arrow(ax, 2.5, 4.4, 4.5, 3.5)
create_arrow(ax, 7.5, 4.4, 5.5, 3.5)

# Capa 3: Dropout Externo
create_box(ax, 3, 2.7, 4, 0.8, 'Dropout Externo\nrate = 0.2 (o 0.3)', color_dropout, fontsize=10)
create_arrow(ax, 5, 2.7, 5, 2.0)

# Capa 4: Dense
create_box(ax, 3, 1.2, 4, 0.8, 'Dense Layer\n3 unidades (softmax)\nNegativo | Neutro | Positivo', color_dense, fontsize=10)
create_arrow(ax, 5, 1.2, 5, 0.5)

# Capa 5: Output
create_box(ax, 3, 0, 4, 0.5, 'Output\nProbabilidades [P(neg), P(neu), P(pos)]', color_output, fontsize=9)

# Anotaciones laterales
ax.text(0.2, 10.7, 'Representación\nVectorial', ha='left', va='center',
        fontsize=9, style='italic', color='gray')
ax.text(0.2, 6.5, 'Procesamiento\nSecuencial', ha='left', va='center',
        fontsize=9, style='italic', color='gray')
ax.text(0.2, 2.3, 'Regularización', ha='left', va='center',
        fontsize=9, style='italic', color='gray')
ax.text(0.2, 0.9, 'Clasificación', ha='left', va='center',
        fontsize=9, style='italic', color='gray')

# Notas al pie
notes = [
    "Nota 1: cuDNN habilitado para LSTM/GRU (dropout interno = 0)",
    "Nota 2: Bidireccionalidad duplica parámetros y tiempo de cómputo",
    "Nota 3: Optimización: Adam (lr=5e-4), class_weights, EarlyStopping"
]
for i, note in enumerate(notes):
    ax.text(5, -0.8 - i*0.3, note, ha='center', va='top',
            fontsize=8, style='italic', color='#555555')

plt.tight_layout()

# Guardar figura
output_dir = 'reports/phase2/figuras'
plt.savefig(f'{output_dir}/fig09_arquitectura_modelo.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/fig09_arquitectura_modelo.pdf', bbox_inches='tight')
print(f"✓ Figura guardada: {output_dir}/fig09_arquitectura_modelo.png")
print(f"✓ Figura guardada: {output_dir}/fig09_arquitectura_modelo.pdf")

plt.close()

