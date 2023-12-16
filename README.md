
# Scalable Duplicate Detection - CS Assignment 501247sh

Date: 15-12-2023

This algorithm implements the MSMPE duplicate detection algorithm as proposed by Stijn van den Heuvel, student number 501247sh, as part of the 2023 Computer Science for Business Analytics Assignment at Erasmus University Rotterdam. This algorithm forms and extension to the MSMP+ algorithm, a LSH-based model-words-driven method. 


## Acknowledgements
The algorithm was based on an extension of the following works
 - van Bezu, R., Borst, S., Rijkse, R., Verhagen, J., Frasincar, F., Vandic, D.: Multi- component similarity method for web product duplicate detection. In: 30th Sympo- sium on Applied Computing. pp. 761–768. ACM (2015)
 - van Dam, I., van Ginkel, G., Kuipers, W., Nijenhuis, N., Vandic, D., Frasincar, F.: Duplicate detection in web shops using lsh to reduce the number of computations. In: 31th ACM Symposium on of Applied Computing. pp. 772–779. ACM (2016)
 - Hartveld, A., van Keulen, M., Mathol, D., van Noort, T., Plaatsman, T., Frasin- car, F., Schouten, K.: An lsh-based model-words-driven product duplicate detection method. In: 30th International Conference on Advanced Information Systems En- gineering). pp. 149–161. Springer (2015)

And applied knowledge from the Computer Science for Business Analytics course taught by Dr. Flavius Frasincar. Special thanks to Dr. Frasincar for his weekly support. 


## Usage

The duplicate detection algorithm uses as inputs the unique list of TV brands (can initial list can easily be found online) as specified in TV_brands.txt. 

The data set on which the algorithm is applied is TVs-all-merged.json, which shows product information of 1624 TVs from four Web shops, www.amazon.com, www.bestbuy.com, www.thenerds.net and www.newegg.com.

The algorithm specified in main.py can perform several specific functions separately within the duplicate deteciton algorithm, which can be activated using the True/False statements from line 1062 to 1066.

Set 'clean_data_now' to True if you want to input the original data sets and apply data cleaning and normalization to save a cleaned_normalized_dataset.json dataset and a cleaned_normalized_extension.json dataset which applies proposed MSMPE extensions. 

Set 'printer' to True if you want the algorithm to print several statements useful for debugging. 

Set 'run_bootstraps_now' to True if you want to apply bootstrapping to the three specified algorithms, MSMP+, MSMP+ clean, and MSMPE. This uses as inputs cleaned_normalized_dataset.json and cleaned_normalized_extension.json and outputs duplicate detection performance metrics and fractions of comparisons per bootstrap per level of t to files to be saved in a 'results' directory, with separate files for each algorithm.

Set 'plot_now' to True if you want to use create several plots using the output files of 'run_bootstraps_now' from the 'results' directory. These plots are saved to a 'newest_plots' directory. 

Set 'calc_matrices' to True if you want to pre-calculate dissimilarity_matrices for the entire dataset for all specified values of alpha, beta, gamma and mu parameters in the gridsearch. Only recommended for optimization and evaluation purposed of this smaller dataset. If you want to run this algorithm on a larger dataset, use the function get_dissimilarity_matrix directly on only the candidate pairs for the given dataset. It uses as inputs the cleaned_normalized_dataset and cleaned_normalized_extension datasets. These matrices are output to a 'Matrices' directory per combination of alpha, beta, gamma and mu per given dataset (specified by data_name_clean and data_name_extension). 
```

