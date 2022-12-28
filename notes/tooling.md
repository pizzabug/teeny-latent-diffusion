# Tooling notes

For study, some notes on tooling while iterating through the model in WSL.

## CUDA Prerequisites

CUDA needs to be installed. For WSL, PATH needs to be defined to point towards the custom driver. 

```
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```