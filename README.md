# Va3DE
Va3DE: Variational 3D Embedding for Single-Cell Hi-C Clustering

![va3de architecture](https://github.com/dylan-plummer/Va3DE/blob/main/assets/images/model.png)

To run Va3DE download and install the package:

```
git clone https://github.com/dylan-plummer/Va3DE.git;
cd Va3DE;
pip install .
```

Then you just need to supply a `.scool` file and a reference file with cell metadata:

```
va3de --scool data/pfc.scool --reference data/pfc_ref 
```
