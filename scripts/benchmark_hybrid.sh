#!/bin/bash
# Benchmark das versões híbridas - CORRIGIDO para mostrar SSE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_DIR="$PROJECT_ROOT/data"
MAX_ITER=50
EPS=1e-6

echo "=== Benchmark Híbridos ==="
echo ""

# OpenMP + CUDA
if [ -f "$PROJECT_ROOT/hybrid/kmeans_1d_omp_cuda" ]; then
    echo " OpenMP + CUDA"
    echo ""
    
    for THREADS in 2 4; do
        for BS in 256; do
            echo "  Threads: $THREADS | Block size: $BS | Dataset: MÉDIO"
            OMP_NUM_THREADS=$THREADS "$PROJECT_ROOT/hybrid/kmeans_1d_omp_cuda" \
                "$DATA_DIR/dados_medio.csv" \
                "$DATA_DIR/dados_medio_centroides_init.csv" \
                $MAX_ITER $EPS $THREADS $BS 2>&1 | \
                grep -E "(K-means 1D \(Híbrido: OpenMP|Threads OpenMP|Iterações|Tempo)"
            echo ""
        done
    done
    echo "---"
    echo ""
fi

# OpenMP + MPI
if [ -f "$PROJECT_ROOT/hybrid/kmeans_1d_omp_mpi" ]; then
    echo " OpenMP + MPI"
    echo ""
    
    for PROCS in 2 4; do
        for THREADS in 2 4; do
            echo "  Processos: $PROCS | Threads: $THREADS | Total: $((PROCS*THREADS)) | Dataset: MÉDIO"
            mpirun -np $PROCS "$PROJECT_ROOT/hybrid/kmeans_1d_omp_mpi" \
                "$DATA_DIR/dados_medio.csv" \
                "$DATA_DIR/dados_medio_centroides_init.csv" \
                $MAX_ITER $EPS $THREADS 2>&1 | \
                grep -E "(K-means 1D \(Híbrido: OpenMP|Processos|Iterações|Tempo)"
            echo ""
        done
    done
    echo "---"
    echo ""
fi

# MPI + CUDA
if [ -f "$PROJECT_ROOT/hybrid/kmeans_1d_mpi_cuda" ]; then
    echo " MPI + CUDA"
    echo ""
    
    for PROCS in 2 4; do
        for BS in 256; do
            echo "  Processos: $PROCS | Block size: $BS | Dataset: GRANDE"
            mpirun -np $PROCS "$PROJECT_ROOT/hybrid/kmeans_1d_mpi_cuda" \
                "$DATA_DIR/dados_grande.csv" \
                "$DATA_DIR/dados_grande_centroides_init.csv" \
                $MAX_ITER $EPS $BS 2>&1 | \
                grep -E "(K-means 1D \(Híbrido: MPI|Processos MPI|Iterações|Tempo)"
            echo ""
        done
    done
    echo "---"
    echo ""
fi

echo "==========================="
