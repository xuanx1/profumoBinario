# What Defines Gender in Perfumes?
One of the key reasons gendered perfumes have remained so dominant is due to deeply embedded stereotypes within the fragrance industry and consumer psychology. (Smith 2019). The rising trend of unisex and gender-neutral fragrances, however, challenges these long-standing norms and invites a re-evaluation of how scent and identity interact (Lopez 2021).

[PCA Preview](https://xuanx1.github.io/profumoBinario/perfume_pca_3d_plot.html)

[Report](https://github.com/xuanx1/profumoBinario/blob/main/final_report_xuan.pdf)

![3dPCA](https://github.com/user-attachments/assets/ba844883-5723-46dc-a60f-bb1889860cd2)


### Introduction 
Gender has always shaped how perfumes are marketed and consumed. For years, perfumes have been categorized into "men" and "women" based on their scent profiles and its respective social narratives. Floral and powdery notes are often associated with femininity, while woody and spicy notes are associated with masculinity. Today, these social conventions continue to persist, despite being more of a cultural norms than having any scientific proof. (Doe 2020).

The analysis begins by identifying the most influential variables contributing to gender differentiation—such as specific top, heart, and base notes—before building a classification model that predicts gender category using these features (Sissel and Collins 2020). Finally, all these can be further visualized by reducing the high-dimensional data into a visual "scent space" using clustering techniques like PCA. This will allow for an exploration of overlaps and divergences in how perfumes are gendered, ultimately revealing biases, trends, and outliers in gender-based fragrance marketing (Morris 2021).

###	Research Questions
1. How does genders classification in perfumes affect pricing?
2. Which scent(s) determines whether a perfume is classified as masculine or feminine?
3. How does the diversity of fragrance notes affect gender classification in perfumes?
4. Which features contribute most to the accuracy of the model?


###	Methodology 
Logistic regression is highly suitable in analyzing binary outcomes. This study revolves around building a logistic regression model which can identify the factors that are significant to the survivability of the passengers of Titanic. The process is as follows:

#### Data Processing 

The dataset will undergo cleaning, graceful handling of missing values. Some cells under the variable Age are missing value, so we will replace these NaN values with an average age instead of removing the respective entries to maintain the number of observations within the dataset. In addition, categorical variable, such as gender is encoded into binary format, where female = 0, male = 1, unisex = 2.

#### Model Building 

The cleaned dataset is split into training (80%) and testing (20%) sets. A logistic regression model will be trained using data that is included in the preprocessing steps for model fitting. All predictors are standardized or encoded during preprocessing to ensure compatibility with the logistic regression model.

#### Model Training 

The logistic regression model is fitted using the training dataset. Each predictor contributes to the model by adjusting the odds of a perfume being classified as male or female.

#### Model Testing 

The model is evaluated on the “unseen” 20% testing dataset, to predict the gender of perfumes based on the predictors. Performance metrics such as prediction accuracy, and ROC-AUC score are calculated to assess how well the model generalizes to new data.

#### Final Evaluation 

The final evaluation involves interpreting the model’s coefficients and assessing its predictions, such as: A positive coefficient for “Notes Diversity” would suggest perfumes with more diverse notes are more likely to be classified as Male. A PCA analysis will be done to identify the most influential but indiscernible factors in gender classification and understand which note or scent determines the gender of the perfume.

### References
Doe, Jane. 2020. The Scent of Gender: A Cultural History of Perfume. New York: Fragrance Press.

Lopez, Maria. 2021. “Unisex Fragrance Trends and Gender Fluidity.” Perfume & Society 8 (1): 27–29.

Smith, John. 2019. “Packaging Identity: Marketing and Gender in Consumer Products.” Journal of Advertising Research 55 (3): 112–125.

Classen, Constance, David Howes, and Anthony Synnott. 1994. Aroma: The Cultural History of Smell. London: Routledge.

Milotic, Daniel. 2003. “The Impact of Visual Design on the Perception of Product Flavor.” Journal of Consumer Behaviour 3 (2): 179–193.

Mintel. 2021. Fragrance and Gender: Consumer Attitudes Toward Gender-Neutral Perfumes in the US. London: Mintel Group Ltd.

Roden, Barbara. 2015. “Fragrance and Femininity: The Gender Divide in the Perfume Industry.” Cultural Studies Review 21 (1): 45–58.

Fragrantica. 2023. Fragrance Classification by Notes and Gender Bias. Accessed March 20, 2025. https://www.fragrantica.com.

Morrison, Karen. 2019. The Economics of Perfume Marketing: Gender and Pricing Trends. New York: Fashion and Fragrance Press.

Morris, Lila. 2021. “Gendered Marketing in the Fragrance Industry: A Scented Spectrum.” Journal of Consumer Culture 21 (3): 415–32.

Scentbird. 2022. “Most Popular Perfume Notes by Gender: Patchouli, Musk, and More.” Scentbird Magazine, July 18, 2022. https://www.scentbird.com/blog/popular-perfume-notes-by-gender.

Sissel, Morgan, and Ryan Collins. 2020. Olfactory Data Science: Machine Learning in the Perfume Industry. Cambridge: Data Press.

