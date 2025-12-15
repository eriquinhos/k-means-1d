# K-means 1D - Projeto de Programação Concorrente e Distribuída

**Implementação completa do algoritmo K-means para dados unidimensionais com múltiplos paradigmas de paralelização: Serial, OpenMP, CUDA, MPI e abordagens híbridas.**

---

## 📋 Sumário

- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Início Rápido](#início-rápido)
- [Uso Detalhado](#uso-detalhado)
- [Formato de Dados](#formato-de-dados)
- [Parâmetros](#parâmetros)
- [Análise de Resultados](#análise-de-resultados)
- [Resultados Esperados](#resultados-esperados)
- [Troubleshooting](#troubleshooting)
- [Contribuindo](#contribuindo)
- [Referências](#referências)

---

## 🎯 Visão Geral

Este projeto implementa o **algoritmo K-means 1D** (Lloyd, 1982) com foco em **análise comparativa de paralelização**. Trata-se de um trabalho acadêmico para a disciplina **Programação Concorrente e Distribuída** que explora:

- **Baseline Serial**: Implementação sequencial para comparação
- **OpenMP**: Paralelização em memória compartilhada (multi-thread)
- **CUDA**: Aceleração em GPU (computação massivamente paralela)
- **MPI**: Computação distribuída (multi-nó)
- **Híbridas**: Combinações de paradigmas para máxima eficiência

### Complexidade Temporal

| Versão | Complexidade | Observações |
|--------|-------------|-------------|
| Serial | O(q·k·n) | q=iterações, k=clusters, n=pontos |
| OpenMP | O(q·k·n/t) | t=threads (speedup teórico) |
| CUDA | O(q·k·n/b) | Transferência GPU overhead |
| MPI | O(q·k·n/p) + comunicação | p=processos, latência de rede |

---

## 📁 Estrutura do Projeto

```
projeto-pcd-kmeans/
├── serial/                      # Baseline: implementação sequencial
│   ├── Makefile
│   ├── kmeans_1d_naive.c
│   └── README.md
│
├── openmp/                      # Paralelização com OpenMP
│   ├── Makefile
│   ├── kmeans_1d_omp.c
│   └── README.md
│
├── cuda/                        # Aceleração com CUDA
│   ├── Makefile
│   ├── kmeans_1d_cuda.cu
│   └── README.md
│
├── mpi/                         # Computação distribuída com MPI
│   ├── Makefile
│   ├── kmeans_1d_mpi.c
│   └── README.md
│
├── hybrid/                      # Implementações híbridas
│   ├── Makefile
│   ├── kmeans_1d_omp_cuda.cu    # OpenMP + CUDA
│   ├── kmeans_1d_omp_mpi.c      # OpenMP + MPI
│   ├── kmeans_1d_mpi_cuda.cu    # MPI + CUDA
│   └── README.md
│
├── data/                        # Geração e armazenamento de datasets
│   ├── generate_data.py         # Script Python para gerar dados sintéticos
│   ├── dados_pequeno.csv        # 10k pontos, 4 clusters
│   ├── dados_medio.csv          # 100k pontos, 8 clusters
│   └── dados_grande.csv         # 1M pontos, 16 clusters
│
├── scripts/                     # Utilitários de benchmark
│   ├── benchmark_all.sh         # Executa todos os benchmarks
│   ├── benchmark_serial.sh
│   ├── benchmark_openmp.sh
│   ├── benchmark_cuda.sh
│   ├── benchmark_mpi.sh
│   └── benchmark_hybrid.sh
│
├── analysis/                    # Análise e visualização
│   ├── analyze_results.py       # Script Python para análise
│   ├── plot_speedup.py
│   ├── plot_scaling.py
│   └── plot_efficiency.py
│
├── report/                      # Saída: gráficos e relatórios
│   ├── figures/                 # Gráficos gerados
│   │   ├── speedup_*.png
│   │   ├── scaling_*.png
│   │   └── efficiency_*.png
│   └── RESULTS.md               # Análise detalhada
│
├── results/                     # Dados brutos de benchmark
│   ├── serial_results.csv
│   ├── openmp_results.csv
│   ├── cuda_results.csv
│   ├── mpi_results.csv
│   └── hybrid_results.csv
│
├── Makefile                     # Orquestração geral
├── README.md                    # Este arquivo
└── LICENSE                      # Licença do projeto
```

---

## 📦 Requisitos

### Compiladores e Ferramentas

```bash
# GCC 7.0+ com C99 (obrigatório)
gcc --version

# OpenMP (geralmente incluído no GCC)
gcc -fopenmp --version

# CUDA Toolkit 11.0+ (opcional, para GPU)
nvcc --version
nvidia-smi

# Open MPI 4.0+ (opcional, para distribuído)
mpicc --version
mpirun --version

# Python 3.8+ (opcional, para análises)
python3 --version
pip3 install matplotlib numpy
```

### Verificação Rápida

```bash
# Rodar script de validação
chmod +x scripts/check_setup.sh
./scripts/check_setup.sh
```

### Instalação de Dependências

#### Ubuntu/Debian
```bash
# GCC e OpenMP
sudo apt-get install build-essential gcc g++ gomp

# OpenMP dev (se necessário)
sudo apt-get install libomp-dev

# Open MPI
sudo apt-get install openmpi-bin libopenmpi-dev

# Python
sudo apt-get install python3 python3-pip
pip3 install matplotlib numpy scipy
```

#### CentOS/RHEL
```bash
# GCC com OpenMP
sudo yum install gcc gcc-c++ gomp

# Open MPI
sudo yum install openmpi openmpi-devel
export PATH=$PATH:/usr/lib64/openmpi/bin

# Python
sudo yum install python3 python3-pip
pip3 install matplotlib numpy scipy
```

#### macOS (Homebrew)
```bash
# GCC com OpenMP
brew install gcc

# Open MPI
brew install open-mpi

# Python
brew install python3
pip3 install matplotlib numpy scipy
```

---

## 🚀 Início Rápido

### 1️⃣ Clonar e Entrar no Diretório

```bash
cd projeto-pcd-kmeans
```

### 2️⃣ Compilar Todas as Versões Disponíveis

```bash
make compile-all
```

Isso detecta automaticamente qual compilador está disponível e compila apenas as versões suportadas.

### 3️⃣ Gerar Dados de Teste

```bash
make data
```

Gera datasets sintéticos em `data/`:
- `dados_pequeno.csv`: 10k pontos, 4 clusters
- `dados_medio.csv`: 100k pontos, 8 clusters
- `dados_grande.csv`: 1M pontos, 16 clusters

### 4️⃣ Executar um Exemplo Rápido

```bash
cd serial
make
./kmeans_1d_naive ../data/dados_pequeno.csv ../data/dados_pequeno_centroides.csv 50 1e-6
```

### 5️⃣ Rodar Todos os Benchmarks

```bash
make benchmark-all
```

Resulta em `results/*.csv` com timings.

### 6️⃣ Gerar Análises e Gráficos

```bash
make analyze
```

Produz gráficos em `report/figures/` (requer Python).

---

## 🔧 Uso Detalhado

### Serial (Baseline)

```bash
cd serial && make

./kmeans_1d_naive <dados.csv> <centroides_init.csv> <max_iter> <eps>
```

**Exemplo:**
```bash
./kmeans_1d_naive ../data/dados_medio.csv ../data/dados_medio_centroides_init.csv 50 1e-6
```

**Saída:**
```
Iteração 1/50: SSE = 45321.23, Δ = 1.000000
Iteração 2/50: SSE = 22451.12, Δ = 0.505682
...
Convergência em iteração 18
Tempo total: 0.342s
```

### OpenMP (Multi-thread)

```bash
cd openmp && make

./kmeans_1d_omp <dados.csv> <centroides_init.csv> <max_iter> <eps> [threads] [schedule] [chunk]
```

**Parâmetros OpenMP:**
- `threads`: Número de threads (0 = detectar automaticamente)
- `schedule`: Estratégia de divisão (`static`, `dynamic`, `guided`)
- `chunk`: Tamanho do chunk (0 = automático)

**Exemplo com 8 threads e static scheduling:**
```bash
./kmeans_1d_omp ../data/dados_medio.csv ../data/dados_medio_centroides_init.csv 50 1e-6 8 static 0
```

**Esperado:**
```
Threads OpenMP: 8
Scheduling: static
Iteração 1/50: SSE = 45321.23, Δ = 1.000000 [tempo par: 0.012s]
...
Speedup: ~7-8x vs serial
```

### CUDA (GPU)

```bash
cd cuda && make

./kmeans_1d_cuda <dados.csv> <centroides_init.csv> <max_iter> <eps> [block_size]
```

**Parâmetros CUDA:**
- `block_size`: Threads por bloco (128, 256, 512)

**Exemplo com 256 threads/bloco:**
```bash
./kmeans_1d_cuda ../data/dados_grande.csv ../data/dados_grande_centroides_init.csv 50 1e-6 256
```

**Esperado:**
```
GPU: NVIDIA ...
Block size: 256
Memória GPU: ... MB
Iteração 1/50: SSE = 45321.23, Δ = 1.000000
Tempo H2D (cópia host→device): 0.045s
Tempo D2H (cópia device→host): 0.051s
Speedup: ~10-12x vs serial (dataset grande)
```

### MPI (Distribuído)

```bash
cd mpi && make

mpirun -np <num_processos> ./kmeans_1d_mpi <dados.csv> <centroides_init.csv> <max_iter> <eps>
```

**Exemplo com 4 processos:**
```bash
mpirun -np 4 ./kmeans_1d_mpi ../data/dados_grande.csv ../data/dados_grande_centroides_init.csv 50 1e-6
```

**Esperado:**
```
MPI processos: 4
Rank 0: Lendo dados...
Rank 1,2,3: Distribuindo chunk...
Iteração 1/50: SSE = 45321.23
Sincronização inter-processos: 0.003s
Speedup: ~4x vs serial (overhead de comunicação)
```

### Híbridos

#### OpenMP + CUDA

```bash
cd hybrid && make

./kmeans_1d_omp_cuda <dados.csv> <centroides_init.csv> <max_iter> <eps> <threads> <block_size>
```

**Exemplo:** 4 threads OpenMP + 256 threads CUDA/bloco
```bash
./kmeans_1d_omp_cuda ../data/dados_medio.csv ../data/dados_medio_centroides_init.csv 50 1e-6 4 256
```

#### OpenMP + MPI

```bash
mpirun -np 2 ./kmeans_1d_omp_mpi <dados.csv> <centroides_init.csv> <max_iter> <eps> [threads]
```

**Exemplo:** 2 processos MPI + 4 threads/processo
```bash
mpirun -np 2 ./kmeans_1d_omp_mpi ../data/dados_grande.csv ../data/dados_grande_centroides_init.csv 50 1e-6 4
```

#### MPI + CUDA

```bash
mpirun -np 4 ./kmeans_1d_mpi_cuda <dados.csv> <centroides_init.csv> <max_iter> <eps> [block_size]
```

**Exemplo:** 4 processos MPI, cada com GPU
```bash
mpirun -np 4 ./kmeans_1d_mpi_cuda ../data/dados_grande.csv ../data/dados_grande_centroides_init.csv 50 1e-6 256
```

---

## 📄 Formato de Dados

### Arquivo CSV de Dados

**Requisitos:**
- **Sem cabeçalho**
- **Um valor por linha** (1D)
- Delimitadores suportados: vírgula, ponto-e-vírgula, espaço, tabulação
- Valores numéricos (float ou int)

**Exemplo (`dados.csv`):**
```
10.5
23.1
15.7
8.2
19.4
...
```

### Arquivo CSV de Centróides Iniciais

**Requisitos:**
- **Sem cabeçalho**
- **Um centróide por linha** (valores iniciais)
- Mesmo delimitador que dados

**Exemplo (`centroides_init.csv`):**
```
10.0
20.0
15.0
5.0
```

### Gerar Dados Sintéticos

```bash
cd data
python3 generate_data.py --points 100000 --clusters 8 --output dados_custom.csv --seed 42
```

**Opções:**
- `--points`: Número de pontos (padrão: 100000)
- `--clusters`: Número de clusters (padrão: 8)
- `--output`: Arquivo de saída (padrão: dados.csv)
- `--seed`: Seed para reprodutibilidade (padrão: 42)
- `--range`: Intervalo de valores (padrão: 0-1000)

---

## ⚙️ Parâmetros

### Parâmetros Comuns

| Parâmetro | Descrição | Tipo | Padrão | Intervalo |
|-----------|-----------|------|--------|-----------|
| `max_iter` | Máximo de iterações | int | 50 | 1-1000 |
| `eps` | Critério de convergência (variação relativa SSE) | float | 1e-4 | 1e-8 a 1e-2 |

### Parâmetros OpenMP

| Parâmetro | Descrição | Tipo | Padrão | Opções |
|-----------|-----------|------|--------|--------|
| `threads` | Número de threads | int | 0 (auto) | 1-64 |
| `schedule` | Divisão de trabalho | string | static | `static`, `dynamic`, `guided` |
| `chunk` | Tamanho do bloco | int | 0 (auto) | 0-1000 |

**Recomendações de scheduling:**
- `static`: Balanceamento predefinido, baixo overhead (bom para loops regulares)
- `dynamic`: Balanceamento runtime, alto overhead (bom para carga desigual)
- `guided`: Híbrido (chunks grandes no início, pequenos no fim)

### Parâmetros CUDA

| Parâmetro | Descrição | Tipo | Padrão | Recomendado |
|-----------|-----------|------|--------|-----------|
| `block_size` | Threads por bloco | int | 256 | 128-512 |

**Trade-offs de block_size:**
- **128**: Menos occupancy, mais registros livres
- **256**: Balanço (recomendado)
- **512**: Máxima occupancy, menos registros/thread

### Parâmetros MPI

| Parâmetro | Descrição | Tipo | Recomendado |
|-----------|-----------|------|-----------|
| `-np` | Número de processos | int | ≤ nós da máquina |

---

## 📊 Análise de Resultados

### Scripts de Análise

```bash
make analyze  # Roda todas as análises
```

Ou individualmente:

```bash
cd analysis

# Gráficos de speedup
python3 plot_speedup.py

# Escalabilidade (strong/weak scaling)
python3 plot_scaling.py

# Eficiência paralela
python3 plot_efficiency.py

# Relatório completo
python3 analyze_results.py
```

### Outputs Gerados

1. **Speedup** (`report/figures/speedup_*.png`)
   - Gráfico: Speedup vs número de recursos
   - Compara todas as versões paralelizadas vs serial
   - Mostra limite teórico (Lei de Amdahl)

2. **Escalabilidade** (`report/figures/*_scaling.png`)
   - Strong scaling: Problema fixo, aumentar recursos
   - Weak scaling: Aumentar problema e recursos proporcionalmente
   - Identifica ponto de saturação

3. **Eficiência** (`report/figures/efficiency_*.png`)
   - Eficiência paralela: Speedup / num_recursos
   - Esperado: 70-90% para bom scaling

4. **Relatório Completo** (`report/RESULTS.md`)
   - Análise estatística
   - Recomendações de uso
   - Trade-offs

### Interpretação de Resultados

```
Speedup Linear (ideal):     S(p) = p  →  Eficiência = 100%
Speedup Sublinear (real):   S(p) < p  →  Eficiência < 100% (overhead)
Lei de Amdahl:              S(p) = 1 / [f + (1-f)/p]
                            (f = fração serial)
```

---

## 📈 Resultados Esperados

### Comparação de Speedup vs Serial

| Versão | Dataset Pequeno (10k) | Dataset Médio (100k) | Dataset Grande (1M) |
|--------|---------------------|----------------------|-------------------|
| **Serial** | 1x (baseline) | 1x (baseline) | 1x (baseline) |
| **OpenMP (8 threads)** | 6-7x | 7-8x | 7-8x |
| **CUDA (256 blocks)** | 5-8x* | 10-12x | 12-15x |
| **MPI (4 processos)** | 2-3x** | 3.5-4x | 4-4.5x |
| **OpenMP + CUDA** | 8-10x | 12-14x | 15-18x |
| **OpenMP + MPI** | 6-8x | 8-10x | 10-12x |
| **MPI + CUDA** | 10-12x | 15-18x | 18-22x |

**Observações:**
- *Dataset pequeno: overhead CUDA domina (requer ≥100k pontos)
- **MPI tem overhead de comunicação; melhor em datasets massivos

### Quando Usar Cada Paradigma

| Paradigma | Caso de Uso | Dataset | Hardware |
|-----------|-----------|---------|----------|
| **Serial** | Debugging, baseline, teórico | Qualquer | CPU |
| **OpenMP** | Workstation multi-core, compartilhado | Médio-Grande | CPU multi-core |
| **CUDA** | GPU disponível, cálculo intensivo | Grande (>100k) | NVIDIA GPU |
| **MPI** | Cluster distribuído, problema massivo | Massivo (>10M) | Multi-nó |
| **OpenMP+CUDA** | Workstation forte, máxima performance | Grande | CPU + GPU |
| **OpenMP+MPI** | Cluster com nós multi-core | Massivo | Multi-nó multi-core |
| **MPI+CUDA** | Sistema HPC, múltiplas GPUs | Massivo | Multi-GPU cluster |

---

## 🔍 Troubleshooting

### ❌ Compilação GCC falha

**Erro:** `gcc: command not found`

**Solução:**
```bash
# Instalar GCC
sudo apt-get install gcc  # Ubuntu/Debian
sudo yum install gcc      # CentOS/RHEL
brew install gcc          # macOS

# Verificar
gcc --version
```

---

### ❌ OpenMP não encontrado

**Erro:** `-fopenmp: unrecognized command line option`

**Solução:**
```bash
# Instalar OpenMP (geralmente com GCC)
gcc --version  # Se GCC 7+, OpenMP deve estar incluído

# Alternativa (libgomp)
sudo apt-get install libgomp1

# Ou compilar sem OpenMP
make ENABLE_OPENMP=0
```

---

### ❌ CUDA não instalado

**Erro:** `nvcc: command not found`

**Solução:**
```bash
# Baixar CUDA Toolkit 11.0+ de https://developer.nvidia.com/cuda-toolkit
# Instalar seguindo o guia oficial

# Verificar após instalação
nvcc --version
nvidia-smi

# Se não encontrar, adicionar ao PATH
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

---

### ❌ MPI não encontrado

**Erro:** `mpicc: command not found`

**Solução:**
```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin libopenmpi-dev

# CentOS/RHEL
sudo yum install openmpi openmpi-devel
export PATH=$PATH:/usr/lib64/openmpi/bin

# Verificar
mpicc --version
mpirun --version
```

---

### ❌ Erro de memória GPU

**Erro:** `CUDA error: out of memory`

**Solução:**
```bash
# Usar dataset menor
./kmeans_1d_cuda data/dados_pequeno.csv centroides.csv 50 1e-6 256

# Ou reduzir block_size
./kmeans_1d_cuda data/dados_medio.csv centroides.csv 50 1e-6 128

# Verificar memória GPU disponível
nvidia-smi

# Limpar GPU
nvidia-smi --query-gpu=memory.free --format=csv
```

---

### ❌ Python não encontrado (análises)

**Erro:** `ModuleNotFoundError: No module named 'matplotlib'`

**Solução:**
```bash
# Instalar Python 3.8+
python3 --version

# Instalar dependências
pip3 install matplotlib numpy scipy

# Ou usar requirements.txt (se existir)
pip3 install -r requirements.txt
```

---

### ❌ MPI falha em executar

**Erro:** `No hosts file was found` ou problema de conectividade

**Solução:**
```bash
# Para execução local (sem cluster real)
mpirun --allow-run-as-root -np 4 ./kmeans_1d_mpi ...

# Ou especificar localhost
mpirun -H localhost -np 4 ./kmeans_1d_mpi ...
```

---

### ❌ Makefile não funciona

**Erro:** `make: *** No targets specified. Stop.`

**Solução:**
```bash
# Verificar Makefile existe
ls -la Makefile

# Executar alvo específico
make compile-all
make benchmark-all

# Forçar recompilação
make clean
make compile-all
```

---

## 🧹 Limpeza

```bash
# Remove executáveis compilados
make clean

# Remove executáveis + dados gerados
make distclean

# Remove tudo (executáveis, dados, resultados, gráficos)
make full-clean
```

---

## 🤝 Contribuindo

Para adicionar novas implementações ou otimizações:

### Passos

1. **Crie um diretório** com nome descritivo
   ```bash
   mkdir new_paradigm/
   cd new_paradigm/
   ```

2. **Adicione Makefile** seguindo padrão do projeto
   ```makefile
   CC = gcc
   CFLAGS = -O3 -std=c99
   
   all: kmeans_1d_new
   
   kmeans_1d_new: kmeans_1d_new.c
   	$(CC) $(CFLAGS) -o kmeans_1d_new kmeans_1d_new.c
   
   clean:
   	rm -f kmeans_1d_new
   ```

3. **Mantenha interface CLI consistente**
   - Argumentos: `<dados.csv> <centroides.csv> <max_iter> <eps> [params...]`
   - Saída: SSE por iteração em formato consistente

4. **Adicione README.md** documentando particularidades

5. **Crie script de benchmark** em `scripts/benchmark_<paradigm>.sh`

6. **Atualize Makefile raiz** e este README

### Template de README para novo paradigma

```markdown
# K-means 1D - [Nome do Paradigma]

## Descrição
[Breve descrição técnica]

## Compilação
[Instruções específicas]

## Uso
[Exemplos e parâmetros]

## Requisitos
[Dependências específicas]

## Notas de Implementação
[Decisões de design, trade-offs]
```

---

## 📚 Referências

### Artigos Científicos

1. **Lloyd, S.** (1982). "Least squares quantization in PCM." IEEE Transactions on Information Theory, 28(2), 129-137.
   - Artigo original do algoritmo K-means

2. **Wang, H., et al.** (2011). "Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic Programming." The R Journal, 3(2), 29-33.
   - Análise aprofundada de K-means 1D e otimizações

3. **OpenMP Architecture Review Board** (2021). "OpenMP API Specification 5.1."
   - Especificação completa de OpenMP

4. **NVIDIA Corporation** (2023). "CUDA C Programming Guide."
   - Guia oficial de programação CUDA

5. **Message Passing Interface Forum** (2021). "MPI: A Message-Passing Interface Standard 3.1."
   - Especificação MPI completa

### Recursos Online

- [GCC Compiler](https://gcc.gnu.org/)
- [OpenMP Official](https://www.openmp.org/)
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-toolkit)
- [Open MPI Project](https://www.open-mpi.org/)
- [Python Scientific Stack](https://www.scipy.org/)

---

## 📄 Licença

Este projeto é material educacional para a disciplina **Programação Concorrente e Distribuída** da [Instituição/Universidade].

Você é livre para estudar, modificar e distribuir este código para fins educacionais. Para uso comercial, consulte a licença completa em `LICENSE`.
---

## 📝 Notas Finais

### Dicas de Uso

1. **Comece pelo serial** para entender o baseline
2. **Use datasets pequenos** para debug, grandes para análise final
3. **Mantenha consistência** de convergência entre versões (validate!)
4. **Documente seus temings** com `time` ou `perf`
5. **Reproduza resultados** com seeds fixas

### Próximos Passos

- [ ] Implementar validação de resultados (checksum dos centróides)
- [ ] Adicionar profiling com `gprof` ou `perf`
- [ ] Estender para K-means 2D/3D
- [ ] Integrar com bibliotecas (OpenCL, TensorFlow)
- [ ] Criar documentação técnica detalhada
- [ ] Adicionar testes unitários

---

**Última atualização:** Dezembro 2025
**Status do Projeto:** Ativo (desenvolvimento educacional)
