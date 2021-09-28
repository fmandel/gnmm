# gnmm
Generalized Neural Network Mixed Model

## Data
The data used in this paper come from a small mobile health pilot study with 15 subjects diagnosed with schizophrenia. Over a 90-day period, data from GPS, accelerometer, call and text logs, screen on/off status, and phone battery were collected almost continuously along with survey questions concerning mood, anxiety, sleep, psychotic symptoms, and general functioning collected a few times a week. Daily summary measures are provided.

## Code
This folder contains all necessary files to replicate the results of the simulation studies in Section 4 and the analysis of the schizophrenia data in Section 5. The contents are:  
* ./Data/feature_matrix_anomalies_trunc3.csv: data from pilot study on patients with schizophrenia
* ./Data/SZ_data_dictionary.xlsx: data dictionary for feature_matrix_anomalies_trunc3.csv
* .R/Network_Functions.R: main functions for fitting GNMM model
* .R/Analysis_of_SZ_data.Rmd: R Markdown file to replicate Table 1 and Figure 4, requires data file and Network_Functions.R
* .R/AUC_LinearFixedEffect_5nodes.R: script to run simulation with binary outcome, linear fixed effect, and 5 hidden nodes and reproduce bottom left panel in Figure 3
* .R/AUC_LinearFixedEffect_10nodes.R: script to run simulation with binary outcome, linear fixed effect, and 10 hidden nodes and reproduce bottom right panel in Figure 3
* .R/AUC_NonlinearFixedEffect_5nodes.R: script to run simulation with binary outcome, nonlinear fixed effect, and 5 hidden nodes and reproduce top left panel in Figure 3
* .R/AUC_NonlinearFixedEffect_10nodes.R: script to run simulation with binary outcome, nonlinear fixed effect, and 10 hidden nodes and reproduce top right panel in Figure 3
* .R/MSPE_LinearFixedEffect_5nodes.R: script to run simulation with continuous outcome, linear fixed effect, and 5 hidden nodes and reproduce bottom left panel in Figure 2
* .R/MSPE_LinearFixedEffect_10nodes.R: script to run simulation with continuous outcome, linear fixed effect, and 10 hidden nodes and reproduce bottom right panel in Figure 2
* .R/MSPE_NonlinearFixedEffect_5nodes.R: script to run simulation with continuous outcome, nonlinear fixed effect, and 5 hidden nodes and reproduce top left panel in Figure 2
* .R/MSPE_NonlinearFixedEffect_10nodes.R: script to run simulation with continuous outcome, nonlinear fixed effect, and 10 hidden nodes and reproduce top right panel in Figure 2

The eight script files for replicating the results of the simulation studies are written for parallel processing using the foreach, doSNOW, and doRNG libraries. The number of cores can be specified by changing the value of the ncores variable. Depending on the number of cores specified, these files can take more than a day to run. Analysis of the schizophrenia data may take several hours to run.
