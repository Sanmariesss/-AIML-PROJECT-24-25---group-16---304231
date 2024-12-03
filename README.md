# AI&ML PROJECT24/25-Group16-304231
# Aeropolis
Group16:
captain - 304231, member 2 - 305681, member 3 - 304121
# Introduction
In the city of Aeropolis, delivery drones are essential for transporting goods quickly and efficiently. However, their cargo capacity (Cargo_Capacity_kg) can be affected by many factors, like weather conditions, terrain type, and the quality of their equipment. The primary aim of this project is to build and train a model that predicts the cargo capacity of drones based on multiple environmental and operational factors. These implemented changes would help us to improve delivery efficiency and gain a deeper understanding of impacts of drone performance.

# Methods
- Describe key ideas for data analysis and modeling. Update as you proceed.

# Experimental Design
- Placeholder for experimental setup (model choices, metrics).

# Results
## Main Findings
The final results indicate that the Linear Regression model performed the best among the models tested, demonstrating the highest R² score of 0.70 and the lowest RMSE of 0.88 on the test set. The R² score suggests that the model explains 70% of the variance in cargo capacity, indicating a strong relationship between the features and the target variable. The RMSE of 0.88 means that, on average, the model’s predictions are off by just 0.88 kg, which is relatively low, indicating that the model makes accurate predictions. These results emphasize the model's reliability in capturing the relationship between the input features and cargo capacity, while maintaining computational efficiency.
### Model Performance Comparison
The table below summarizes the performance of the three models evaluated, due to its combination of the highest R² score and the lowest RMSE.
| Model               | R²   | RMSE  |
|---------------------|-------|-------|
| Linear Regression   | 0.70  | 0.88  |
| Random Forest       | 0.69  | 0.89  |
| Gradient Boosting   | 0.69  | 0.88  |

- **R² (R-squared)** indicates how well the model explains the variance in the target variable. A higher R² means the model captures more of the underlying patterns, so we aim for the highest R² possible.
- **RMSE (Root Mean Squared Error)** measures the average error between predicted and actual values. A lower RMSE means the model makes more accurate predictions.

By selecting the highest R² and lowest RMSE, we ensure the model both fits the data well and provides accurate predictions.

### Results Visualization 
This scatter plot is based on the predictions from the Linear Regression model compared to the actual test set values. The purpose of creating this plot was to see how well the model's predictions align with reality. The red dashed line represents perfect predictions, where the predicted values match the actual ones. 


![True vs Predicted Cargo Capacity](images/true_vs_predicted_cargo_capacity.png)

#### Observations From Scatter Plot:

- **Overall Accuracy**:  
  Most points are clustered around the red line, indicating that the model performs well for the majority of predictions.
- **Extreme Values**:  
  There are noticeable deviations at the higher and lower ends of cargo capacities, suggesting the model struggles with extreme cases.
- **Error Distribution**:  
  The scatter around the line appears relatively consistent, meaning the prediction errors are fairly evenly distributed and do not show strong systematic bias.

This visualization makes it easier to spot patterns, like how the model handles typical values well but struggles a bit with extreme cases.
# Conclusions
- Placeholder for final takeaways and future directions.

