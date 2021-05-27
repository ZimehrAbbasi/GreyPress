# LinearPress

This program takes in an image and compresses it in the same extension. Using Single Value Decomposition(SVD) and the Gram-Schmidt Algorithm, this program is able to reduce the final size in a few seconds.

#### Inputs

The inputs to this program are a `file directory` (absolute or relative path), and a `decomposition percentage`(DP) which is basically how high you want the resolution to be.

### Sample output

These are sample outputs of an RGB image for various `decomposition percentage`s.

<img src="images/rgb.png" width=300 align=right>

| DP  |                        Image                        |
| :-: | :-------------------------------------------------: |
|  1  |  <img src="images/rgb1.png" width=300 align=right>  |
|  5  |  <img src="images/rgb5.png" width=300 align=right>  |
| 10  | <img src="images/rgb10.png" width=300 align=right>  |
| 25  | <img src="images/rgb25.png" width=300 align=right>  |
| 50  | <img src="images/rgb50.png" width=300 align=right>  |
| 100 | <img src="images/rgb100.png" width=300 align=right> |
