# LTP_Final

## Introduction
For the course "Language technology project" at the RUG (academic year 20/21) we compared recurrent neural networks and transformers
in their ability to correctly classify russian troll tweets into 5 distinct categories, which were originally described in the paper by [Linvill & Warren](http://pwarren.people.clemson.edu/Linvill_Warren_TrollFactory.pdf). The data-set was taken from [Kaggle](https://www.kaggle.com/fivethirtyeight/russian-troll-tweets). Our results are described in a final report.

## Requirements
The python libraries outlined in the requirements_conda.txt document are the minimum requirements to replicate our findings.
We relied on R version 4.0.5. Only base R is required to generate the data-files. However, recreating the word clouds requires
the "wordcloud" package (2.6), the "tm" package (0.7-8), and the "RColorBrewer" package (1.1-2) also listed in the [tutorial](https://towardsdatascience.com/create-a-word-cloud-with-r-bde3e7422e8a) which includes the original code used to generate the clouds here as well.

## Setup
To replicate our code first download the original data-set from Kaggle and place the individual IRA* .csv files in the data-folder. Then:
- Use the R-script in the helpers folder to generate the necessary train, dev, and test sets.
- Subsequently, use the train_* files to train and fine-tune the individual models discussed in the report. Start with the train_embeddings file.
- Finally, use the test_* notebooks to evaluate the models on the test-set.
- The embedding_analysis notebook in Results/embedding_results can also be used to inspect the learned embeddings.
