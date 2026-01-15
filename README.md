# PrivateIrisByCKKS
Private Iris Recognition with High-Performance FHE
This repository contains binaries for experiments in the paper `Private Iris Recognition with High-Performance FHE'.

It requires RTX-5090 to run the experiment correctly.

## Getting Started
### 1. Set up conda environment
Create the conda environment (for Linux).
```
conda env create -f conda/iris-ckks.yml
```

Activate the environment.
```
conda activate iris-ckks
```

### 2. Build encrypted DB generator and run it
Move to [db-generator](db-generator), create a build directory, and compile:
```
cd db-generator
mkdir build
cd build
cmake ..
make -j
```

Run the binary to generate random iris data and encrypted database.
```
./worldcoin-db-generation
```

<!-- Check [db-generator/README.md](db-generator/README.md) for the details. -->

### 3. Build private iris recogniction circuits and run them
Move to [iris-ckks](iris-ckks), create a build directory, and compile:
```
cd ../../iris-ckks
mkdir build
cd build
cmake ..
make -j
```

After compilation, you can run each component of the priviate iris circuit.
```
./preProcess
./coreCircuit
./postProcess
```

<!-- Check [iris-ckks/README.md](iris-ckks/README.md) for the details of their outputs. -->
