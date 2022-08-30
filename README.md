# speechbrain

## Model activation extraction
To extract model activations from multiple sound files (and multiple models), use run_models.py. 

To run random network: In the run_models.py script, it is possible to generate randomly permuted tensors for the supported architectures. Set the variable rand_netw to True in the beginning of the script. The permuted architecture will be loaded and model activations will be saved with an appended "_randnetw".

## SepFormer
### Changes to the original model
The ReLU function changed to inplace=True in the encoder part of the model (allows extraction of the post-ReLU Conv activations using the Conv forward hook). The ReLU is specified in forward() (line 193) in /speechbrain/lobes/models/dual_path.py.

## metricGAN
### Changes to the original model
The LeakyReLU was changed to inplace=True in the generator part of the model (allows extraction of the post-LeakyReLU first Linear activations using the Linear forward hook). The LeakyReLU is specified in foward() (line 93) in /speechbrain/lobes/models/MetricGAN.py.

## Acknowledgements
Huggingface and SpeechBrain
