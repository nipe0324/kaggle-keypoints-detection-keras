# kaggle-keypoints-detection-keras

kaggle keypoints detection tutorial by keras with tensorflow


## Execution time

- AWS GPU g2.2xlarge (※ CPUの場合はx2倍ぐらいの時間がかかる)
- CUDA 8.0
- cuDNN 5.0
- TensorFlow 0.1.2
- Keras 0.0.1
- Python 3.5.2 (pyenv)

| model  |      description    |         epoch          |   time   | train loss |  val loss |
|--------|---------------------|------------------------|----------|------------|-----------|
| model1 | NN                  |   100                  |    33sec |    0.00229 |   0.00333 |
| model2 | CNN                 | 1,000                  |    51min |    0.00054 |   0.00168 |
| model3 | model2 + flip       | 3,000                  |   155min |    0.00011 |   0.00160 |
| model4 | model3 + lr         | 1,000                  |    52min |    0.00031 |   0.00158 |
| model5 | model4 + dropout    | 1,000                  |    58min |    0.00169 |   0.00158 |
| model6 | model5 + epoch/fc   | 5,000                  |   341min |    0.00066 |   0.00097 |
| model7 | model6 + early stop | 5,000 (3578:early stop)|   246min |    0.00066 |   0.00094 |
| model8 |                     |                        |          |            |           |

※2〜やる
