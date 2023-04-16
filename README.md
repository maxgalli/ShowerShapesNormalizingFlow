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

where the branch used in ```flows4flows``` is ```dev```.

## Preprocessing

## Training

## Config logs

- 21: preprocessing is performed with pt scaled with the custom transform and then also scaled with all the others, i.e. NOT separately
- 22: we use quantile transform to scale 
- 22B: same as 22, but I screwed up the way to make final dataframes so I rerun it
- 22C: retry the custom context
- 22D: keep everything as 22 but introduce an L2 norm for the middle flow
- 22E: L2 norm also for vertical flows + dropout probability
- 23: add dropout probability everywhere and use qtgaus
- 23B: a bit more dropout
- 23X: same as 23B, but switch data and mc as input to FFFCustom even if it does not make sense
- 24: retry standard scaler
- 24B: same as 24 but save every epoch

Logs from 22 are the ones after changing ParquetDataset to accept functions for scaling and scaling back