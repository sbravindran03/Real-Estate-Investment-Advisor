# EDA Findings

Dataset: 250,000 cleaned listings, 42 cities, 20 states. Good-investment share: 36.7%.

### Q1. What is the distribution of property prices?

Prices range from ₹10L to ₹500L, mean ₹254.6L, median ₹253.9L, skew 0.01 — close to uniform, so there is no long luxury tail.

![q01_price_distribution](eda/q01_price_distribution.png)

### Q2. What is the distribution of property sizes?

Sizes range from 500 to 5000 sq ft (median 2747, skew 0.00).

![q02_size_distribution](eda/q02_size_distribution.png)

### Q3. How does the price per sq ft vary by property type?

Median ₹/sq ft: Villa ₹9,190, Apartment ₹9,245, Independent House ₹9,305. The differences are small — property type alone does not drive unit price.

![q03_ppsf_by_type](eda/q03_ppsf_by_type.png)

### Q4. Is there a relationship between property size and price?

Pearson r = -0.003: size and total price are essentially independent in this dataset, which is why price per sq ft varies so widely.

![q04_size_vs_price](eda/q04_size_vs_price.png)

### Q5. Are there any outliers in price per sq ft or property size?

IQR rule flags 19,723 rows (7.89%) on price per sq ft (small homes priced high) and 0 on size. They are kept (they are real listings) but flagged; tree models are robust to them.

![q05_outliers](eda/q05_outliers.png)

### Q6. What is the average price per sq ft by state?

Highest: Karnataka (₹13,252); lowest: Punjab (₹12,931).

![q06_ppsf_by_state](eda/q06_ppsf_by_state.png)

### Q7. What is the average property price by city?

Most expensive: Bangalore (₹258.5L); cheapest: Cuttack (₹250.8L). Spread across 42 cities is only 3.1%.

![q07_price_by_city](eda/q07_price_by_city.png)

### Q8. What is the median age of properties by locality?

Across 500 localities the median age ranges 16–21 years (overall median 18).

![q08_age_by_locality](eda/q08_age_by_locality.png)

### Q9. How is BHK distributed across cities?

Each BHK level (1–5) makes up roughly 20% of listings in every city — the mix is uniform.

![q09_bhk_by_city](eda/q09_bhk_by_city.png)

### Q10. What are the price trends for the top 5 most expensive localities?

Top 5 by ₹/sq ft: Locality_207, Locality_416, Locality_246, Locality_359, Locality_387. Prices fluctuate by build year without a monotonic trend, so build year is a weak price signal.

![q10_top5_locality_trends](eda/q10_top5_locality_trends.png)

### Q11. How are numeric features correlated with each other?

Strongest pairs: Price_in_Lakhs–Future_Price_5Y (0.99); Size_in_SqFt–Price_per_SqFt (-0.61); Nearby_Schools–Infrastructure_Score (0.59); Nearby_Hospitals–Infrastructure_Score (0.59); Transport_Score–Infrastructure_Score (0.56). Raw features are otherwise nearly uncorrelated.

![q11_correlation](eda/q11_correlation.png)

### Q12. How do nearby schools relate to price per sq ft?

Correlation r = -0.000; the average stays within ₹12,888–₹13,292, so schools counts barely move unit prices.

![q12_schools_vs_ppsf](eda/q12_schools_vs_ppsf.png)

### Q13. How do nearby hospitals relate to price per sq ft?

Correlation r = -0.000; the average stays within ₹12,917–₹13,257, so hospitals counts barely move unit prices.

![q13_hospitals_vs_ppsf](eda/q13_hospitals_vs_ppsf.png)

### Q14. How does price vary by furnished status?

Mean price: Furnished ₹254.4L, Semi-furnished ₹254.3L, Unfurnished ₹255.0L.

![q14_price_by_furnishing](eda/q14_price_by_furnishing.png)

### Q15. How does price per sq ft vary by property facing direction?

Mean ₹/sq ft: East ₹13,024, North ₹13,025, South ₹13,044, West ₹13,140.

![q15_ppsf_by_facing](eda/q15_ppsf_by_facing.png)

### Q16. How many properties belong to each owner type?

Broker: 83,479, Owner: 83,268, Builder: 83,253.

![q16_owner_type](eda/q16_owner_type.png)

### Q17. How many properties are available under each availability status?

Under_Construction: 125,035, Ready_to_Move: 124,965.

![q17_availability](eda/q17_availability.png)

### Q18. Does parking space affect property price?

Mean price: No ₹254.4L, Yes ₹254.7L — negligible difference.

![q18_parking](eda/q18_parking.png)

### Q19. How do amenities affect price per sq ft?

Mean ₹/sq ft by amenity count: 1: ₹13,015, 2: ₹13,059, 3: ₹13,044, 4: ₹13,071, 5: ₹13,102. Amenities matter more for appreciation (and our investment score) than for current price.

![q19_amenities](eda/q19_amenities.png)

### Q20. How does public transport accessibility relate to price per sq ft or investment potential?

- Low: ₹13,086/sq ft, 28.2% good investments, 8.38% expected growth
- Medium: ₹13,062/sq ft, 37.6% good investments, 8.60% expected growth
- High: ₹13,027/sq ft, 44.4% good investments, 8.83% expected growth

![q20_transport](eda/q20_transport.png)

### Extra. Price trends by city & infrastructure vs resale value

Fastest expected appreciation: Bilaspur, Trivandrum, Jamshedpur. Infrastructure score correlates r = 0.23 with growth rate. Size vs 5-year return: r = 0.007. (The dataset has no crime-rate column, so that brief question cannot be answered.)

![x1_growth_infrastructure](eda/x1_growth_infrastructure.png)
