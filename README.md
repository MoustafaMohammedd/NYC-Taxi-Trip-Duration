## Project Overview
This project aims to predict the total ride duration of taxi trips in New York City as part of the NYC Taxi Duration Prediction competition on Kaggle. The model uses data provided by the NYC Taxi and Limousine Commission, including information such as pickup time, geo-coordinates, number of passengers, and other variables.
## Data exploration

### Target Variable: Trip Duration
- Distribution resembles a Gaussian distribution with a long right tail (right-skewed)
- Most trips are between 150 seconds and 1000 seconds (about 2.5 to 16.7 minutes)
- Log transformation applied to visualize better and help with modeling large values

### Feature Analysis
1. Discrete Numerical Features:
   - Vendor ID and passenger count analyzed
   - No significant difference in trip duration among vendors
   - Trips with 7-8 passengers tend to have shorter durations, possibly due to trip purpose

2. Geographical Features:
   - Haversine distance calculated using pickup and dropoff coordinates
   - Most trips range from less than 1 km to 25 km

3. Temporal Analysis:
   - Longer trip durations observed during summer months
   - Weekend trips generally longer than weekdays
   - Shorter durations during morning and evening rush hours

## Modeling

### Data Pipeline
1. Feature splitting into categorical and numerical
2. One-hot encoding for categorical features
3. Standard scaling for numerical features
4. Polynomial Features (degree=4)
5. Log transformation applied

### Results
1. Ridge
      - RMSE: 0.4478 (Validation)
      - R²: 0.6867 (Validation)
2. XGBoost
      - RMSE: 0.4012 (Validation)
      - R²: 0.7485 (Validation)
        
### Lessons and Future Work
- Feature selection improves model performance
- Outlier removal for intra-trip duration doesn't improve performance
- Estimating speed as a separate feature didn't significantly improve results
- Consider exploring more complex algorithms like XGBoost and ensemble methods for better performance
