# LinearPress

This program takes in an image and compresses it in the same extension. Using Single Value Decomposition(SVD) and the Gram-Schmidt Algorithm, this program is able to reduce the image size in a few seconds.

#### Inputs

The inputs to this program are a `file directory` (absolute or relative path), and a `resolution percentage`(RP) which is basically how high you want the resolution to be.

### Sample output

These are sample outputs of an RGB image for various `resolution percentage's`.

![image](images/rgb512.png){: style="float: left"}

| Resolution Percentage | Eigenvalues |                        Image                        |
| :-------------------: | :---------: | :-------------------------------------------------: |
|      0.1953125%       |      1      |  <img src="images/rgb1.png" width=300 align=right>  |
|      0.9765625%       |      5      |  <img src="images/rgb5.png" width=300 align=right>  |
|       1.953125%       |     10      | <img src="images/rgb10.png" width=300 align=right>  |
|      4.8828125%       |     25      | <img src="images/rgb25.png" width=300 align=right>  |
|       9.765625%       |     50      | <img src="images/rgb50.png" width=300 align=right>  |
|       19.53125%       |     100     | <img src="images/rgb100.png" width=300 align=right> |
|       39.0625%        |     200     | <img src="images/rgb200.png" width=300 align=right> |
|         100%          |     512     | <img src="images/rgb512.png" width=300 align=right> |
