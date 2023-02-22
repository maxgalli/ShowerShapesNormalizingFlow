# Shower Shapes Corrections with Normalizing Flows

## Setup 

For development, which means installing in editable mode also ```nflows``` and ```flows4flows```:

```
cd packages_to_install
git clone git@github.com:maxgalli/nflows.git
git clone git@github.com:maxgalli/flows4flows.git
mamba env create -f environment_minimal.yml
conda activate FFF-minimal
cd packages_to_install/nflows
pip install -e .
cd packages_to_install/flows4flows
pip install -e .
```

where the branch used in ```flows4flows``` id ```dev```.