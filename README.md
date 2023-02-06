# VanillaGCN
Implementation of a VanillaGCN for the case of a Digital Twin
 
First, you will need to install python and pip
##Docker

If you want to make new graph data, you will need to download Docker. Follow the documentation at:
- Docker Desktop: https://docs.docker.com/desktop/
- Docker Engine: https://docs.docker.com/engine

## Setting up the enviroment  

For using all code included in the BCN Challenge, in case you want to create new data:
```
pip install notebook==6.4.11 PyYAML==6.0 tensorflow==2.7 networkx==2.8.1 matplotlib==3.5.2 astropy==5.1
```
If you only want to use the model with the preloaded data:

```
pip install pytorch
pip install pyg
```

I highly recommend to make both installations because some parts of the code cannot works without above installed libraries.
