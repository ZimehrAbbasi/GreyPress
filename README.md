# Compresser

This program takes in an image and compresses it in the same extension. Using Single Value Decomposition(SVD) and the Gram-Schmidt Algorithm, this program is able to reduce the final size in a few seconds.

#### Inputs

The inputs to this program are a `file directory` (absolute or relative path), and a `decomposition percentage`(DP) which is basically how high you want the resolution to be.

#### Samples

![Original image](images/lion.png)

| DP  |          Image          |
| :-: | :---------------------: |
|  1  |  ![](images/lion1.png)  |
| 10  | ![](images/lion10.png)  |
| 20  | ![](images/lion20.png)  |
| 50  | ![](images/lion50.png)  |
| 100 | ![](images/lion100.png) |
| 200 | ![](images/lion200.png) |
| 400 | ![](images/lion400.png) |
