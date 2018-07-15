# medical-diagnosis
Medical Diagnosis using Patient’s Notes

crawler.py - 
scrape the mtsamples site and download data from relevant medical classes

txtTocsv.py -
convert the raw file data to csv with a column containing the full description from the medical notes

clampxmlparser.py -
parse the xml output from clamp to csv containing columns for problem, drug, treatment, test

NB_SVM_EntireDescription.py -
file to take csv with whole description as an input and train & test our model to predict the medical speciality for the record

NB_SVM_ClampData.py - 
file to take csv made from clamp data as an input and train our model to predict the medical speciality for the record

run_saved_models.py -
runs the test file against the model saved by using the clamp data and gives the score of accuracy for the same
