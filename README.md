# Ease My Trip Flight Price Prediction

This project aims to leverage an extensive dataset of flight booking records sourced from the "Ease My Trip" website to develop predictive models for estimating ticket prices. With a focus on incorporating various influencing factors such as airline choice, travel class, booking lead time, and other significant attributes, our goal is to create a robust analytical framework that enables stakeholders to gain insights into flight pricing dynamics.

## Introduction

The aviation industry is a highly dynamic and competitive sector that is characterized by fluctuating ticket prices, varying demand patterns, and a multitude of influencing factors. As a result, airlines and travel agencies are constantly seeking innovative solutions to optimize pricing strategies, enhance revenue management practices, and improve customer satisfaction levels.

## Project Goal

The primary objective of this project is to develop machine learning models that can accurately predict flight ticket prices based on historical booking data. By analyzing the relationship between ticket prices and various attributes such as airline, travel class, booking lead time, and other relevant factors, we aim to create predictive models that can assist stakeholders in making informed decisions regarding pricing, inventory management, and customer segmentation.

## The Data

The dataset used in this project consists of historical flight booking records obtained from the "Ease My Trip" website. The dataset contains information about various attributes such as airline, travel class, booking lead time, departure and arrival locations, and ticket prices. By leveraging this dataset, we can explore the underlying patterns and trends in flight pricing and develop predictive models that can estimate ticket prices with a high degree of accuracy.

###### For information about the data snapshot, refer to this project [link](https://gitlab.com./aelluminate/databank/2024-10/ease-my-trip-flight-booking).

## Methodology

The project will follow a structured methodology that involves the following key steps:

- **Data Preprocessing**: The dataset will be cleaned, transformed, and prepared for analysis. This step involves handling missing values, encoding categorical variables, and normalizing numerical features.
- **Exploratory Data Analysis (EDA)**: We will conduct a comprehensive analysis of the dataset to identify patterns, trends, and relationships between different attributes. This step will involve visualizations, statistical summaries, and correlation analysis.
- **Visualizations**: We will create visualizations such as scatter plots, histograms, and box plots to gain insights into the distribution of ticket prices and other attributes.
- **Model Development**: We will develop machine learning models using regression techniques to predict flight ticket prices. The models will be trained on historical booking data and evaluated using appropriate performance metrics.
- **Model Evaluation**: We will evaluate the performance of the predictive models using metrics such as Mean Absolute Error (MAE) and R-squared value. This step will help us assess the accuracy and reliability of the models.

## Tools

The project will be implemented using Python programming language and popular libraries such as **Pandas**, **NumPy**, **Seaborn** **Matplotlib**, and **Scikit-learn**. These libraries provide robust tools for data manipulation, visualization, and machine learning model development.

## Visualization

### Price Distribution

![Price Distribution](https://i.imgur.com/SH6GNfI.png)

> It illustrates the relationship between flight prices and their corresponding counts. The histogram's x-axis represents flight prices in Indian Rupees (₹), ranging from 0 to 120,000, while the y-axis indicates the frequency of flights at each price level.    
> 
> The graph reveals that a substantial number of flights fall within the lower price range, particularly around 0 to 10,000 in Rupees (₹), where the highest frequency exceeds 100,000 flights. This trend indicates that lower prices attract more flight options. As the price increases, the number of available flights decreases significantly, creating a sharp decline in flight counts after the initial peak, eventually plateauing around 60,000 ₹.
>
> Additionally, the overlayed smooth line (Kernel Density Estimate, KDE) further emphasizes the decreasing density of flights as prices rise. The analysis provides insight into flight pricing trends, emphasizing a clustering of affordable options compared to higher prices, which results in fewer available flights.

### Price Distribution by Class

![Price Distribution by Class](https://i.imgur.com/0GmnSrI.png)

> The price distribution of tickets is illustrated through a box plot comparing two classes: **Economy** and **Business**.
>
> The **Economy class** box shows a lower median price range, with its lower whisker extending to approximately ₹0, indicating the minimum price within the dataset. The median for economy tickets is represented by a line within the box, slightly above the bottom quartile, and several outliers are visible above the upper whisker, suggesting some tickets are priced significantly higher.
>
> In contrast, the **Business class** box appears taller and broader, confirming a higher median price and a broader range of prices. Its median line is centered in the box, indicating a noticeably higher price point than the Economy class. Outliers in the Business class are represented by dots above and below the whiskers, indicating a greater variance in ticket pricing.

### Price vs Duration

![Price vs Duration](https://i.imgur.com/80E3RqF.png)

> The trend line indicates a general positive correlation between price and duration, suggesting that as flight duration increases, prices also tend to rise. However, the data also showcases a range of outliers, indicating that similar durations can exhibit significant price variations.
>
> The overall distribution reflects a diminished density of points in the middle and upper sections of the plot, signifying that longer flights tend to have a wider array of prices and may not always ensure higher costs.

### Price Distribution by Airline

![Price Distribution by Airline](https://i.imgur.com/WrU3pKr.png)

> - **Price Distribution by Airline**: It illustrates the range and distribution of flight prices for different airlines. Each box denotes the interquartile range (IQR), with the median price indicated by a line within the box. Distinct outliers can be observed above the upper whiskers for certain airlines. For instance, SpiceJet and AirAsia exhibit a narrower price range, signifying less variability in their pricing, while Vistara displays a broader price spectrum with higher median prices, suggesting more expensive flight options.
>
> - **Average Price by Airline**: It complements this data by comparing the average flight prices across airlines. Vistara stands out with the highest average price, markedly exceeding its competitors, indicating a trend toward higher ticket costs. In contrast, SpiceJet and AirAsia maintain lower average prices, reflected by their shorter bars, which may appeal to cost-conscious travelers.

### Departure Time Analysis

![Departure Time Analysis](https://i.imgur.com/9PUOlz8.png)

> - The tallest bars are observed in the Morning and Evening categories for the Economy class, both approaching the maximum flight count of nearly 50,000. This suggests that these times are particularly busy for Economy flights, indicating a strong demand for travel during these periods.
> - The Early Morning category has a noticeably lower count for both classes, yet Economy class still has a higher representation. In contrast, the Afternoon and Night categories show a significant drop in flight counts, with Business class flights seeing especially reduced activity.
> - The Late Night category exhibits the least flight activity overall, with Economy class recording just less than 10,000 flights and Business class having minimal or close to zero representation, which aligns with typical travel behavior where fewer flights are scheduled during late hours.
>
> Overall, the graph effectively illustrates travel patterns, suggesting that more passengers prefer Economy flights during peak hours, while Business class flights see lower numbers throughout the day, particularly during off-peak times.

### Stops Analysis

![Stops Analysis](https://i.imgur.com/KmaKqKV.png)

> - **Stops Distribution**: It indicates that the majority of flights are non-stop or have one stop. Specifically, 83.6% of flights have one stop, while 12.0% are non-stop (zero stops), and only 4.4% of flights have two or more stops. This suggests that most travelers prefer fewer stops when booking flights.
>
> - **Price Distribution by Stops**: It demonstrates significant variations in pricing based on the number of stops. Non-stop flights show a narrow price distribution, indicating that prices are relatively consistent in this category. In contrast, flights with one stop exhibit a wider price range, indicating greater variability in pricing within this segment. Lastly, flights with two or more stops have both a limited range and lower average prices, suggesting they are less frequently offered and less competitively priced.

## Model Evaluation

The predictive models developed in this project will be evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: This metric measures the average absolute difference between the predicted and actual flight ticket prices. A lower MAE indicates better model performance.
- **R-squared Score**: This metric quantifies the proportion of variance in the flight ticket prices that can be explained by the model. A higher R-squared value indicates a better fit between the predicted and actual prices.

The following table summarizes the performance of the predictive models based on the evaluation metrics:

| Model           | Mean Absolute Error (MAE) | R-squared Score |
|-----------------|---------------------------|-----------------|
| **XGBoost**         | **0.01632**                  | **0.976**            |
| Gradient Boost  | 0.02416            | 0.952            |
| Linear Regression | 0.03733           | 0.9113           |
| Ridge Regression | 0.03733            | 0.9113           |

## Conclusion

In conclusion, this project aims to develop predictive models for estimating flight ticket prices based on historical booking data. By leveraging machine learning techniques and analyzing various influencing factors such as airline choice, travel class, booking lead time, and other attributes, we can create robust models that provide accurate price predictions. The visualizations and analysis presented in this project offer valuable insights into flight pricing dynamics and help stakeholders make informed decisions regarding pricing strategies, revenue management, and customer segmentation. The evaluation metrics demonstrate the effectiveness of the predictive models in estimating flight ticket prices with high accuracy, highlighting the potential for practical applications in the aviation industry. 