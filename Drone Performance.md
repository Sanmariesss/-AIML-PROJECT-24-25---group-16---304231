# Drone Cargo Capacity Prediction — Aeropolis Dataset

**Course:** Artificial Intelligence & Machine Learning — Luiss Guido Carli (2024/25)  
**Team:** Sabina Nurseitova, Ziyi Dong  
**Type:** Supervised Regression | Team Project

---

# Introduction

In the fictional city of Aeropolis, autonomous delivery drones are the backbone of urban logistics. This project builds a machine learning pipeline to **predict how much cargo (in kg) a drone can carry per flight**, based on 19 environmental and operational features such as weather conditions, terrain type, wind speed, battery type, and autopilot quality.

Accurate cargo prediction can help optimize delivery scheduling, reduce fuel waste, and improve overall drone fleet efficiency.

---

# Dataset

- **Source:** Aeropolis dataset (provided as part of the AI & ML course)
- **Size:** ~700,000 rows × 20 features
- **Target variable:** `Cargo_Capacity_kg` (continuous — regression problem)
- **Key features include:** `Air_Temperature_Celsius`, `Wind_Speed_kmph`, `Terrain_Type`, `Weather_Status`, `Quantum_Battery`, `Autopilot_Quality_Index`, `Flight_Duration_Minutes`

> Due to hardware constraints, we worked on a **1% random sample (~7,000 rows)** using `data.sample(frac=0.01, random_state=42)` to ensure reproducibility.

---

# Methods
## Tools and Environment
  - **Programming Language**: Python
  - **Libraries**:pandas, matplotlib, seaborn, scikit - learn, numpy.
  - **Environment**: Jupyter Notebook
  - **How to recreate an environment**: Set Up the Conda environment. Use this to recreate it:  conda env export > environment.yml command.

**flowchart that illustrates the steps in our machine learning system**
![flowchart](images/flowchart.png)

## Why Regression?

This is a regression problem because the target variable, **Cargo Capacity (kg)**, is a continuous numerical variable. Classification and clustering do not apply here:
- **Classification**: Used for categorical target variables with discrete classes. Not applicable to a continuous measurement.
- **Clustering**: Groups data points based on similiarity without predefined labels. Not applicable to a continuous measurement. 

## Models

To address the problem while balancing simplicity and complexity, we tested three regression models: 

- **Linear Regression**: Interpretable baseline that establishes a linear relationship between features and the target variable, providing insights into how each feature contributes to the prediction. 
- **Random Forest**: Combines multiple decision trees to improve accuracy and reduce overfitting. It captures non-linear relationships between features and the target variable. 
- **Gradient Boosting**: Builds models sequentially, correcting errors of prior iterations. It is known for strong predictive performance on complex patterns.

## Pipeline Steps

We started with **Data Loading and Initial Inspection** to understand the dataset structure, summary statistics, and missing value counts. Then we moved to **EDA**, followed by preprocessing, model training, and evaluation.

### EDA

We aimed to understand patterns, distributions, and correlations in the data.

**Target variable distribution:**

![distribution_of_cargo_capacity_in_kg](images/distribution_of_cargo_capacity_in_kg.png)

The histogram shows that most drones have moderate cargo capacity, while a few carry significantly less.

**Numerical feature analysis** — boxplots were used to identify outliers across numerical features:

![boxplot_of_route_optimization_per_second](images/boxplot_of_route_optimization_per_second.png)

**Categorical feature analysis** — countplots with value labels were used to observe frequency distributions. All categories were similarly balanced:

![countplot_of_flight_zone](images/countplot_of_flight_zone.png)

Then, after getting some vision, we moved to **EDA(Explanatory Data Analysis)**. 

In EDA, we aimed to understand patterns, distributions and correlations of the data. We created visualizations to identify outliers, general trends and relashionships between feautures and our target variable **Cargo Capacity(kg)**. 
To understand what we are working with, we first wanted to see a target variable distribution. For that reason, we chose to draw histogram using plt disctionary to plot a histogram. 
![distribution_of_cargo_capacity_in_kg](images/distribution_of_cargo_capacity_in_kg.png)
As you can see, the plot shows that the target variable is distributed in the way, where the majority of the drones have moderate cargo capacity (can handle moderate amount of weight), while a few can carry significanly less. 
Then, we decided to **detect outliers** to have a better understanding of the distribution of outliers in our feauture. 

**Numerical Feauture Analysis**:

Outliers were identified visually through boxplots of numerical feautures.This helps us to understand and see the data distribution and decide if the it requires outlier handling. This is one of the examples of boxplots we did. 
![boxplot_of_route_optimization_per_second](images/boxplot_of_route_optimization_per_second.png)

**Categorical Feauture Analysis**:

We used countplots to analyze categorical feautures. It might seem weird that we used it for categories, however, we observed them in terms of frequency. How frequent each of the appear and have an influence on our target value. Since their values are very close to each other, so we added a numerical values on top of each bar, so it will be easier to observe and distinguish them. 
![countplot_of_flight_zone](images/countplot_of_flight_zone.png)
This is one of the examples of countplots we did. Each of the bars have more or less similar frequencies, which shows that there is a balanced distribution of flight zones.

Now, let's take a closer look into **preprocessing** stage.
In order to preprocess the dataset for modeling, we applied the following steps:

**Handling of missing values**:

Numerical feautures: we instructed it to impute missing values using the mean. We chose this approach because it preserves the central tendency of the data without introducing significant bias.
Categorical Feautures: missing values were filled with mode (most frequent value).It ensures that imputed values align with the distribution of the data while maintaining the interpretability of categorical variables.

**Encoding categorical variables**:

We applied One-Hot Encoding to convert categorical feautires into binary indicators because they are essential for machine learning models that require numerical inputs (like our regression models) and also avoids introducing ordinal relationships that may not exist in the data. 

**Scaling numerical feautures**:

StandardScaler was used to standardize numerical features by centering them around zero with a unit variance. It ensures that all feautures are on a comparable scale, which is important for models sensitive to feauture magnitude, such as Linear Regression and Gradient Boosting.

**Setting a data sample**:

our chosen dataset has a training set size of 700000 lines, which is a huge amount of information. To be honest, my laptop could not handle it and stoped working every time I tried to work with a dataset. Therefore, in order to be able to work with it, we decided to work with only 1% of the whole dataset. It ensures the successful completion of the project and saves my laptop from dying.

**Data Splitting**:

We splitted the data into 3 sets: **training (70%), validation (15%) and test(15%)** using randomized splits. This ensures that the models are trained on one dataset, validated on another for hyperparameter tuning, and finally tested on unseen data to evaluate real-world performance.

By performing these preprocessing steps, we ensured that the dataset was well-prepared for model training and evaluation, minimizing potential biases and errors while optimizing model performance.

**Model Training**:

For each model, we integrated preprocessing into a Pipeline, ensuring that all data transformations occurred consistently during training and evaluation. The models were first trained using default hyperparameters, and performance was measured on the validation set.

**Hyperparameter Tuning**:

We performed **cross-validation** to fine-tune model hyperparameters for optimal performance. The tuning process used 2-fold cross-validation and R² as the scoring metric. However, it is important to mention that it was only applied to Random Forest and Gradient Boosting models GridSearchCV to identify the best parameter combinations. Linear Regression did not need it because it has no tunable parameters. We just directly trained and evaluated the model. 

**Model Evaluation**:

We chose R² (Coefficient of Determination), RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) as our metrics for evaluation of each model to identify thhe best one for our project. 
**R² (Coefficient of Determination)**:measure of how well the model explains the variance in the target variable. The higher it is, more effective model is working.
**RMSE (Root Mean Squared Error)**:penalizes larger errors more than smaller ones, making a model sensitive to outliers. This metric is particularly useful for real-world interpretability, as you can better understand how far, on average, the predictions deviate from actual values.



# Experimental Design
## Experiment: Model Comparison and Evaluation

### The Main Purpose:
This experiment aimed to evaluate and compare the performance of different machine learning models (Linear Regression, Random Forest, and Gradient Boosting) in predicting drone cargo capacity using the Aeropolis dataset. The goal was to identify the most accurate model for predicting cargo capacity based on various operational and environmental factors.

### Baselines:
- **Baseline Model**: Linear Regression was chosen as the baseline due to its simplicity and widespread use in regression problems. It provides a straightforward benchmark for comparing more complex models.
- **Comparison Models**: Random Forest and Gradient Boosting were selected as alternative models, given their effectiveness in capturing non-linear relationships and complex patterns in the data.

### Evaluation Metrics:
- **R² (R-squared)**: R² was used to evaluate how well the model explains the variance in cargo capacity. A higher R² indicates a better fit between the model's predictions and the actual values, showing the model's ability to capture the underlying trends.
  
- **RMSE (Root Mean Squared Error)**: RMSE was employed to assess the average error in the model’s predictions. Lower RMSE values suggest that the model is more accurate in its predictions. This metric is particularly useful in evaluating the model's performance relative to the scale of the cargo capacity.

### Summary:
In this experiment, we compared multiple models to identify the best approach for predicting drone cargo capacity. Both R² and RMSE were used as evaluation metrics to assess model accuracy and fit. Linear Regression served as the baseline model, and Random Forest and Gradient Boosting were tested to explore whether more sophisticated models could offer improved performance.




# Results
## Main Findings
The final results indicate that the Linear Regression model performed the best among the models tested, demonstrating the highest R² score of 0.70 and the lowest RMSE of 0.88 on the test set. The R² score suggests that the model explains 70% of the variance in cargo capacity, indicating a strong relationship between the features and the target variable. The RMSE of 0.88 means that, on average, the model’s predictions are off by just 0.88 kg, which is relatively low, indicating that the model makes accurate predictions. These results emphasize the model's reliability in capturing the relationship between the input features and cargo capacity, while maintaining computational efficiency.
### Model Performance Comparison
The table below summarizes the performance of the three models evaluated
| Model               | R²   | RMSE  |
|---------------------|-------|-------|
| Linear Regression   | 0.70  | 0.88  |
| Random Forest       | 0.69  | 0.89  |
| Gradient Boosting   | 0.69  | 0.88  |

By selecting the highest R² and lowest RMSE, we ensure the Linear Regression model both fits the data well and provides accurate predictions.

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
#### Key Takeaways
This project demonstrated the potential of using machine learning models, specifically Linear Regression, to predict drone cargo capacity based on various factors. The model achieved solid performance with an R² of 0.70 and an RMSE of 0.88, showing good predictive accuracy for the majority of test cases, and highlighting the relationship between the various features and cargo capacity. These results underscore the model’s ability to effectively capture the relationship between input features and cargo capacity, also suggest that this approach could be a reliable tool for understanding and predicting drone performance in real-world applications.

#### Unanswered Questions & Future Work
However, some questions are still unanswered. For example, the impact of certain environmental factors like wind speed or temperature could not be fully explored due to limitations in the dataset. Additionally, more complex models or feature engineering might improve prediction accuracy, especially for extreme values. In the future work, we could explore using more advanced techniques such as ensemble methods or neural networks, and further collect data to refine our model and include additional factors that may better explain cargo capacity variability.

## Repository Structure

```
├── main.ipynb          # Full annotated notebook (EDA → modeling → evaluation)
├── aeropolis.csv       # Dataset
├── README.md           # This file
└── images/             # Plots referenced in README
    ├── true_vs_predicted_cargo_capacity.png
    ├── distribution_of_cargo_capacity_in_kg.png
    ├── boxplot_of_route_optimization_per_second.png
    └── countplot_of_flight_zone.png
```
## How to Run

1. Clone the repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open `main.ipynb` in Jupyter Notebook or Google Colab
4. Update the dataset path in cell 2 to your local path
5. Run all cells in order


