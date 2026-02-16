# Prime Zeta Research

Research project exploring the connection between prime numbers and Riemann zeta function zeros.

## Overview

This project analyzes the residual term R(p) in prime number approximations and investigates its spectral characteristics to match with zeta function zeros.

### Main Research Stages

1. **Prime Generation** - Sieve of Eratosthenes
2. **Need(p) and CN(p) Computation** - Auxiliary functions
3. **Spectral Analysis** - FFT and wavelet analysis of the residual term
4. **Zero Matching** - Comparing spectrum peaks with theoretical ζ(s) frequencies

## Requirements

- Python 3.11+
- PostgreSQL (optional, for large datasets)
- Dependencies listed in `pyproject.toml`

## Installation

```bash
# Clone repository
git clone https://github.com/Sett11/prime-zeta-research.git
cd prime-zeta-research

# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

### Configuration

Edit `config.yaml` to customize parameters:

- `max_n` - Maximum number for prime generation
- `c_value` - Window parameter C
- `gamma_k_file` - Path to zeta function zeros file. The file is NOT included in the repo and must be downloaded separately.
  - Download: Get the file from https://oeis.org/A002410 (zeros of zeta function) or a similar source
  - Format: One zero per line (imaginary part gamma_k)
  - Expected filename: `data/gamma_k_5000000.txt` (first 5M zeros)

### Running the Pipeline

```bash
# Stage 0: Prime generation
python scripts/0_generate_primes.py

# Stage 1: Need(p) and CN(p) computation
python scripts/1_compute_need_cn.py

# Stage 2: Spectral analysis (FFT/Wavelet)
python scripts/2_spectral_analysis.py

# Stage 3: Matching with zeta function zeros
python scripts/3_match_zeros.py

# Stage 4: Scalogram analysis
python scripts/4_analyze_scalogram.py

# Stage 5: Save R(p) dynamics
python scripts/5_save_R_dynamics.py

# Stage 6: Detailed FFT comparison with zeros
python scripts/6_compare_fft_with_zeros.py
```

### Optional: PostgreSQL

For large datasets, use PostgreSQL:

```bash
# Start PostgreSQL via docker-compose
docker-compose up -d

# Set engine: postgres in config.yaml
```

## Project Structure

```
prime-zeta-research/
├── config.yaml          # Experiment configuration
├── pyproject.toml       # Project dependencies
├── docker-compose.yml   # PostgreSQL for large data
├── README.md
├── LICENSE
├── scripts/             # Main pipeline scripts
│   ├── 0_generate_primes.py
│   ├── 1_compute_need_cn.py
│   ├── 2_spectral_analysis.py
│   ├── 3_match_zeros.py
│   ├── 4_analyze_scalogram.py
│   ├── 5_save_R_dynamics.py
│   └── 6_compare_fft_with_zeros.py
├── src/                 # Source code
│   ├── config_loader.py
│   ├── primes/          # Prime number generation
│   ├── analysis/        # Analysis (Li, regression, residuals)
│   ├── database/        # Database operations
│   ├── spectral/        # FFT, wavelets
│   └── utils/           # Utilities
└── data/                # Data (in .gitignore)
```

## Results

Results are saved in `data/` directory:
- Plots (PNG)
- Statistics (JSON)
- Binary data (NPZ)

## Scientific Background

### Functions

- **Need(p)** - Auxiliary function for prime analysis
- **CN(p)** - Cumulative sum of Need(p)
- **Li(p)** - Logarithmic integral
- **R(p) = CN(p) - k·Li(p) - b** - Residual term

### Spectral Analysis

- FFT(R) provides the frequency spectrum of the residual term
- Peak frequencies are matched with γ_k/(2π), where γ_k are imaginary parts of ζ(s) zeros

## License

MIT License - See LICENSE file

## Authors

Prime Zeta Research Team
