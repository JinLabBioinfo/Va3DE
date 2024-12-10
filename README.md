# Va3DE
Va3DE: Variational 3D Embedding for Single-Cell Hi-C Clustering

![va3de architecture](https://github.com/JinLabBioinfo/Va3DE/blob/main/assets/images/model.png)

To run Va3DE download and install the package:

```
git clone https://github.com/JinLabBioinfo/Va3DE.git;
cd Va3DE;
pip install .
```

Then you just need to supply a `.scool` file and a reference file with cell metadata:

```
va3de --scool data/pfc.scool --reference data/pfc_ref 
```

You can find a full example here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18hawprSdTEQLeNLiqMrVOFBTqct5BP4q?usp=sharing)

We supply a variety of preprocessed scHi-C datasets and metadata in our SCORE package: https://github.com/JinLabBioinfo/SCORE
