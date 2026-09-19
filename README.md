![Makemore Neural Network](docs/makemore.png)
# Makemore in C++

Implementation of Karpathy's Makemore using LibTorch, PyTorch's c++ backend.

The following architectures are implemented:
- Bigram
- MLP

## Set up (Mac/Linux)

### Building

```bash
# Clone repo
git clone https://github.com/gxddi/makemore-in-c
cd makemore-in-c

# Setup dependencies
./scripts/setup.sh

mkdir build | cd build
cmake -S ../ -B ./
make
```

Requirements:
- git
- cmake
- make
- pip (for setup script)

### Guide
TBA

## Motivations
Provides lower level understanding of pytorch which surprisingly has a C++ backend it binds to.

## Contributions

### AI Contributions
- Heavy contributions to docs/graphs.py


---

#### Made with 🧠
