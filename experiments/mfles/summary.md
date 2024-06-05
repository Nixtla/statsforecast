# Monthly
## Metrics
|       | AutoETS   | AutoMFLES   | AutoTBATS   | DynamicOptimizedTheta   | SeasonalNaive   |
|:------|:----------|:------------|:------------|:------------------------|:----------------|
| mae   | 586.1     | **565.3**   | 591.0       | 577.1                   | 700.2           |
| mape  | 16.9%     | **15.2%**   | 17.0%       | 16.5%                   | 19.2%           |
| mase  | **0.96**  | 0.97        | 1.05        | 0.98                    | 1.26            |
| rmse  | 710.7     | **683.3**   | 737.6       | 697.1                   | 846.7           |
| smape | 6.8%      | **6.5%**    | 6.6%        | 6.7%                    | 8.0%            |

## Times
| model                 |   CPU time (min) |
|:----------------------|-----------------:|
| SeasonalNaive         |                0 |
| DynamicOptimizedTheta |               32 |
| AutoMFLES             |              148 |
| AutoETS               |              445 |
| AutoTBATS             |            6,441 |
# Hourly
## Metrics
|       | AutoARIMA   | AutoETS   | AutoMFLES   | AutoTBATS   | DynamicOptimizedTheta   | SeasonalNaive   |
|:------|:------------|:----------|:------------|:------------|:------------------------|:----------------|
| mae   | 307.8       | 570.8     | **299.6**   | 337.3       | 366.3                   | 353.9           |
| mape  | 16.8%       | 29.3%     | **12.1%**   | 13.9%       | 21.6%                   | 15.6%           |
| mase  | **1.03**    | 1.61      | 1.79        | 1.49        | 2.39                    | 1.19            |
| rmse  | 370.3       | 688.2     | **363.0**   | 400.6       | 458.5                   | 426.3           |
| smape | 6.9%        | 8.6%      | **5.8%**    | 6.3%        | 9.0%                    | 7.0%            |

## Times
| model                 |   CPU time (min) |
|:----------------------|-----------------:|
| SeasonalNaive         |                0 |
| DynamicOptimizedTheta |                1 |
| AutoMFLES             |                7 |
| AutoETS               |               17 |
| AutoTBATS             |              328 |
| AutoARIMA             |            1,233 |
# Weekly
## Metrics
|       | AutoARIMA   | AutoETS   | AutoMFLES   | AutoTBATS   | DynamicOptimizedTheta   | SeasonalNaive   |
|:------|:------------|:----------|:------------|:------------|:------------------------|:----------------|
| mae   | **327.3**   | 332.3     | 338.7       | 353.8       | 332.0                   | 727.2           |
| mape  | **7.5%**    | 8.4%      | 8.3%        | 9.2%        | 8.0%                    | 15.7%           |
| mase  | **0.53**    | 0.57      | 0.57        | 0.56        | 0.55                    | 1.22            |
| rmse  | **397.4**   | 402.5     | 402.1       | 417.6       | 403.7                   | 790.7           |
| smape | **3.8%**    | 4.3%      | 4.0%        | 4.6%        | 3.9%                    | 7.3%            |

## Times
| model                 |   CPU time (min) |
|:----------------------|-----------------:|
| SeasonalNaive         |                0 |
| DynamicOptimizedTheta |                5 |
| AutoMFLES             |               18 |
| AutoETS               |               27 |
| AutoTBATS             |              133 |
| AutoARIMA             |            3,880 |
