To replicate the model used in the paper the code should be run in hte following order
    1. The Scraper
    2. The Tokeniser
    3. The Model
The first 2 steps can be skipped by using the pretokenised data set available at https://doi.org/10.7910/DVN/OHW4OK, the code to make it is for replication purposes.
To replicate the US model you can use the pre tokenised dataset or download the transscript data from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SHFQYU and then run the US tokeniser. Using the code labelled tensormodeltranssciptsUS to run the model.
To run torch adquately a decidicated gpu is recommened.
To avaluate the Australian Data on teh US Model after training the US model you cna run modelevaulate.py
To run the model adquately a decidicated gpu is recommened.
Model results will vary slightly due seemingly torch not having a way of introducing a replicable ramdom seed.