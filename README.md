# Interactive dashboard to visualize estimated age-sex specific life expectancy extension with eliminated deaths attributed to 26 manageable risk factors for 204 countries

* All data kindly provided by Global Burden of Disease Study 2019 - https://vizhub.healthdata.org/gbd-results/
* Jupyter notebook with data preprocessing - https://github.com/NikitiusIvanov/gbd-life-extension-dashboard/blob/main/preprocessing_all_countries.ipynb
* Jupyter notebook witn calculation impact of risk factors into life expectancy - https://github.com/NikitiusIvanov/gbd-life-extension-dashboard/blob/main/life_expectancy_extension_estimation_all_countries.ipynb

In this project we are using following stack:
  * Python as main programming language with libraries and frameworks:
    * Pandas, numpy - to data processing and calculation
    * Plotly Dash - to build web aplication with interactive visualizations
    * Docker - to application contenirization
    * Google cloud Run - to application deploy

![Demo gif](https://github.com/NikitiusIvanov/gbd-life-extension-dashboard/blob/main/demo.gif)
