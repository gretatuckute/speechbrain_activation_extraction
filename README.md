# speechbrain_activation_extraction

## Model activation extraction 
Currently metricGAN and SepFormer are supported.

To extract activations from multiple sound files, use run_models.py. 

To run permuted network: In the run_models.py script, it is possible to generate randomly permuted tensors for the architecture of interest. Set the variable rand_netw to True in the beginning of the script. 
