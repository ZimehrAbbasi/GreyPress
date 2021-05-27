# Compresser

This program takes in an image and compresses it in the same extension. Using Single Value Decomposition(SVD) and the Gram-Schmidt Algorithm, this program is able to reduce the final size in a few seconds.

#### Inputs

The inputs to this program are a `file directory` (absolute or relative path), and a `decomposition percentage`(DP) which is basically how high you want the resolution to be.

#### Samples

![Original image](images/lion.png)

| DP  |                     Image                      |
| :-: | :--------------------------------------------: |
|  1  |  ![Original image](images/lion1.png =400x267)  |
| 10  | ![Original image](images/lion10.png =400x267)  |
| 20  | ![Original image](images/lion20.png =400x267)  |
| 50  | ![Original image](images/lion50.png =400x267)  |
| 100 | ![Original image](images/lion100.png =400x267) |
| 200 | ![Original image](images/lion200.png =400x267) |
| 400 | ![Original image](images/lion400.png =400x267) |
