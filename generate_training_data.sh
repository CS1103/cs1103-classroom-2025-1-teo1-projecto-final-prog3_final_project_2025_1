#!/bin/bash

# Script para generar datos de entrenamiento sintéticos y CSV automáticamente
echo "📊 Generando datos de entrenamiento..."

# Crear directorio de datos si no existe
mkdir -p data

# Generar datos sintéticos para entrenamiento
cat > data/synthetic_training_data.txt << 'EOF'
XOR Data:
0,0 -> 0
0,1 -> 1
1,0 -> 1
1,1 -> 0

Pong Data:
Ball at (0.5,0.5) with velocity (-0.05,0.02)
Paddle at (0.9,0.5) - Action: 0 (Move Left)
Paddle at (0.1,0.5) - Action: 1 (Move Right)

Validation Cases:
0.2,0.6 -> 1
0.7,0.3 -> 0
EOF

# Generar training_data.csv
cat > data/training_data.csv << 'EOF'
ball_x,ball_y,ball_vx,ball_vy,paddle_y,action
0.1,0.5,-0.05,0.02,0.6,0
0.3,0.7,0.03,-0.04,0.4,1
0.8,0.2,0.06,0.01,0.3,2
0.5,0.9,-0.02,0.05,0.8,0
0.9,0.1,0.04,-0.03,0.2,1
0.2,0.6,0.05,0.02,0.4,1
0.7,0.3,-0.03,0.04,0.6,0
0.4,0.8,0.02,-0.05,0.7,2
0.6,0.4,0.01,0.03,0.5,1
0.3,0.1,-0.04,-0.01,0.1,2
EOF

# Generar validation_data.csv
cat > data/validation_data.csv << 'EOF'
ball_x,ball_y,ball_vx,ball_vy,paddle_y,expected_action
0.2,0.6,0.05,0.02,0.4,1
0.7,0.3,-0.03,0.04,0.6,0
0.4,0.8,0.02,-0.05,0.7,2
0.9,0.1,0.06,0.01,0.2,1
0.1,0.9,-0.04,0.03,0.9,0
EOF

echo "✅ Datos generados:"
echo "  - data/synthetic_training_data.txt (XOR, Pong y casos de validación)"
echo "  - data/training_data.csv ($(wc -l < data/training_data.csv) líneas)"
echo "  - data/validation_data.csv ($(wc -l < data/validation_data.csv) líneas)"
echo ""
echo "📊 Muestra de datos de entrenamiento:"
head -5 data/training_data.csv
