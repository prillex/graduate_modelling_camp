# Concise Final Model Comparison

Filtered to `price_sold` with stratified KFold cross-validation.

## Best Run

Random Forest with Full features: test R2 = 0.900 +/- 0.003.

## Runs

| model         | feature_case   |   test_r2_mean |   test_r2_std |   test_rmse_mean |   test_rmse_std |   generalisation_gap_r2 |
|:--------------|:---------------|---------------:|--------------:|-----------------:|----------------:|------------------------:|
| Linear        | Reduced        |         0.6306 |        0.0075 |       87646.0392 |       1170.1517 |                  0.0003 |
| MLP           | Reduced        |         0.7694 |        0.0036 |       69260.9888 |        647.7451 |                  0.0094 |
| Random Forest | Reduced        |         0.8438 |        0.0044 |       56990.4373 |       1075.9608 |                  0.0218 |
| Linear        | Time           |         0.6335 |        0.0074 |       87309.5919 |       1180.0430 |                  0.0003 |
| MLP           | Time           |         0.7702 |        0.0030 |       69143.5137 |        302.3981 |                  0.0143 |
| Random Forest | Time           |         0.8997 |        0.0035 |       45661.9563 |        930.3515 |                  0.0461 |
| Linear        | Full           |         0.7046 |        0.0046 |       78386.3073 |        892.5377 |                  0.0006 |
| MLP           | Full           |         0.8113 |        0.0045 |       62653.8090 |        858.7199 |                  0.0202 |
| Random Forest | Full           |         0.9003 |        0.0034 |       45525.8203 |        940.8482 |                  0.0461 |
