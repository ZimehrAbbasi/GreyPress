# LinearPress

This program takes in an image and compresses it in the same extension. Using Single Value Decomposition(SVD) and the Gram-Schmidt Algorithm, this program is able to reduce the image size in a few seconds.

#### Inputs

The inputs to this program are a `file directory` (absolute or relative path), and a `resolution percentage`(RP) which is basically how high you want the resolution to be.

### Sample output

These are sample outputs of an RGB image for various `resolution percentage's`.

<img src="images/rgb.png" width=300 align=right>

| Resolution Percentage | Eigenvalues |                        Image                        |
| :-------------------: | :---------: | :-------------------------------------------------: |
|      0.1953125%       |      1      |  <img src="images/rgb1.png" width=300 align=right>  |
|      0.9765625%       |      5      |  <img src="images/rgb5.png" width=300 align=right>  |
|       1.953125%       |     10      | <img src="images/rgb10.png" width=300 align=right>  |
|      4.8828125%       |     25      | <img src="images/rgb25.png" width=300 align=right>  |
|       9.765625%       |     50      | <img src="images/rgb50.png" width=300 align=right>  |
|       19.53125%       |     100     | <img src="images/rgb100.png" width=300 align=right> |
|         100%          |     512     | <img src="images/rgb512.png" width=300 align=right> |

### Evaluation

After viewing the images above one cannot tell the slightest difference between the last 3 images even though one is 20% the resolution of the next. This shows us that images can be compressed by a large margin without affecting the quality too much. The next step in the development of this program is too add machine learning capabilities to determine the ideal cutoff for the eigenvalues to use for the compressed image so that the quality is not affected.
