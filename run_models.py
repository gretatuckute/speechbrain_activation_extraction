from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import torch
import random
import numpy as np
from os.path import join
from extractor_utils import SaveOutput
from tqdm import tqdm
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

## SETTINGS ##
rand_netw = False
source_model = 'sepformer'

RESULTDIR = f'/Users/gt/Documents/GitHub/aud-dnn/aud_dnn/model-actv/{source_model}/'
DATADIR = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds_16kHz/'
ROOTDIR = '/Users/gt/Documents/GitHub/speechbrain_extract_actv/'

files = [f for f in os.listdir(DATADIR) if os.path.isfile(os.path.join(DATADIR, f))]
wav_files_identifiers = [f for f in files if f.endswith('wav')]
wav_files_paths = [DATADIR + f for f in wav_files_identifiers]

if __name__ == '__main__':
	##### SEPFORMER #####
	if source_model == 'sepformer':
		model = separator.from_hparams(source="speechbrain/sepformer-whamr",
									   savedir='pretrained_models/sepformer-whamr')
		
		## Get random network ##
		if rand_netw: # Randomize all three relevant modules (encoder, masknet and decoder)
			print('OBS! RANDOM NETWORK!')
			
			## ENCODER ##
			state_dict_encoder = model.modules['encoder'].state_dict()
			state_dict_encoder_rand = {}
			
			if not Path(os.path.join(ROOTDIR,
									 f'{source_model}_encoder_randnetw_indices.pkl')).exists():  # random network indices not generated
				# The following code was used to generate indices for random permutation ##
				d_rand_idx = {}  # create dict for storing the indices for random permutation
				for k, v in state_dict_encoder.items():
					w = state_dict_encoder[k]
					idx = torch.randperm(w.nelement())  # create random indices across all dimensions
					d_rand_idx[k] = idx
				
				with open(os.path.join(ROOTDIR, f'{source_model}_encoder_randnetw_indices.pkl'), 'wb') as f:
					pickle.dump(d_rand_idx, f)
			else:  # load random indices
				d_rand_idx = pickle.load(open(os.path.join(ROOTDIR, f'{source_model}_encoder_randnetw_indices.pkl'), 'rb'))
			
			for k, w in state_dict_encoder.items():
				# Load random indices
				print(f'________ Loading random indices from permuted architecture for {k} ________')
				idx = d_rand_idx[k]
				rand_w = w.view(-1)[idx].view(
					w.size())  # permute using the stored indices, and reshape back to original shape
				state_dict_encoder_rand[k] = rand_w
			
			model.modules['encoder'].load_state_dict(state_dict_encoder_rand)
			
			## MASKNET ##
			state_dict_masknet = model.modules['masknet'].state_dict()
			state_dict_masknet_rand = {}
			
			if not Path(os.path.join(ROOTDIR,
									 f'{source_model}_masknet_randnetw_indices.pkl')).exists():  # random network indices not generated
				# The following code was used to generate indices for random permutation ##
				d_rand_idx = {}  # create dict for storing the indices for random permutation
				for k, v in state_dict_masknet.items():
					w = state_dict_masknet[k]
					idx = torch.randperm(w.nelement())  # create random indices across all dimensions
					d_rand_idx[k] = idx
				
				with open(os.path.join(ROOTDIR, f'{source_model}_masknet_randnetw_indices.pkl'), 'wb') as f:
					pickle.dump(d_rand_idx, f)
			else:  # load random indices
				d_rand_idx = pickle.load(
					open(os.path.join(ROOTDIR, f'{source_model}_masknet_randnetw_indices.pkl'), 'rb'))
			
			for k, w in state_dict_masknet.items():
				# Load random indices
				print(f'________ Loading random indices from permuted architecture for {k} ________')
				idx = d_rand_idx[k]
				rand_w = w.view(-1)[idx].view(
					w.size())  # permute using the stored indices, and reshape back to original shape
				state_dict_masknet_rand[k] = rand_w
			
			model.modules['masknet'].load_state_dict(state_dict_masknet_rand)
			
			## DECODER ##
			state_dict_decoder = model.modules['decoder'].state_dict()
			state_dict_decoder_rand = {}
			
			if not Path(os.path.join(ROOTDIR,
									 f'{source_model}_decoder_randnetw_indices.pkl')).exists():  # random network indices not generated
				# The following code was used to generate indices for random permutation ##
				d_rand_idx = {}  # create dict for storing the indices for random permutation
				for k, v in state_dict_decoder.items():
					w = state_dict_decoder[k]
					idx = torch.randperm(w.nelement())  # create random indices across all dimensions
					d_rand_idx[k] = idx
				
				with open(os.path.join(ROOTDIR, f'{source_model}_decoder_randnetw_indices.pkl'), 'wb') as f:
					pickle.dump(d_rand_idx, f)
			else:  # load random indices
				d_rand_idx = pickle.load(
					open(os.path.join(ROOTDIR, f'{source_model}_decoder_randnetw_indices.pkl'), 'rb'))
			
			for k, w in state_dict_decoder.items():
				# Load random indices
				print(f'________ Loading random indices from permuted architecture for {k} ________')
				idx = d_rand_idx[k]
				rand_w = w.view(-1)[idx].view(
					w.size())  # permute using the stored indices, and reshape back to original shape
				state_dict_decoder_rand[k] = rand_w
			
			model.modules['decoder'].load_state_dict(state_dict_decoder_rand)
		
		## Set in eval mode ##
		model.modules['encoder'].eval()
		model.modules['masknet'].dual_mdl.eval()
		model.modules['decoder'].eval()
		
		
		### LOOP OVER AUDIO FILES ###
		for filename in tqdm(wav_files_paths):
			# filename = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds_16kHz/stim83_cicadas_TESTx10.wav'
			
			# Write hooks for the model
			save_output = SaveOutput(rand_netw=rand_netw)
			
			hook_handles = []
			layer_names_all = []
			layer_names = []
			layer_dual = []
			
			# Grab encoder embedding
			for idx, layer in enumerate(model.modules['encoder'].modules()):
				layer_names_all.append(layer)
				if isinstance(layer, torch.nn.modules.ReLU):  # after the first NN in each block
					handle = layer.register_forward_hook(save_output)
					hook_handles.append(handle)
					layer_names.append(layer)
				if isinstance(layer, torch.nn.modules.Conv1d):  # after the first NN in each block
					handle = layer.register_forward_hook(save_output)
					hook_handles.append(handle)
					layer_names.append(layer)
			
			# Grab transformer "dual-path" linear activations
			for idx, layer in enumerate(model.modules['masknet'].dual_mdl.modules()):
				# print(layer)
				layer_names_all.append(layer)
				if isinstance(layer, torch.nn.modules.Linear):
					if str(layer).startswith('Linear(in_features=1024'):
						handle = layer.register_forward_hook(save_output)
						hook_handles.append(handle)
						layer_names.append(layer)
			
			# for custom file, change path
			est_sources = model.separate_file(path=join(DATADIR, filename))  # it is sampled down to 8khz
			
			## Detach activations
			detached_activations = save_output.detach_activations_sepformer()
			
			# Get identifier (sound file name)
			id1 = filename.split('/')[-1]
			identifier = id1.split('.')[0]
			
			# Store and save activations
			save_output.store_activations(RESULTDIR=RESULTDIR, identifier=identifier)
	
	#### METRICGAN ####
	if source_model == 'metricGAN':
		from speechbrain.pretrained import SpectralMaskEnhancement
		
		enhance_model = SpectralMaskEnhancement.from_hparams(
			source="speechbrain/metricgan-plus-voicebank",
			savedir="pretrained_models/metricgan-plus-voicebank",
		)
		
		## Get random network ##
		if rand_netw:
			state_dict = enhance_model.modules['enhance_model'].state_dict()
			state_dict_rand = {}
			print('OBS! RANDOM NETWORK!')
			
			if not Path(os.path.join(ROOTDIR,
									 f'{source_model}_randnetw_indices.pkl')).exists():  # random network indices not generated
				# The following code was used to generate indices for random permutation ##
				d_rand_idx = {}  # create dict for storing the indices for random permutation
				for k, v in state_dict.items():
					w = state_dict[k]
					idx = torch.randperm(w.nelement())  # create random indices across all dimensions
					d_rand_idx[k] = idx
				
				with open(os.path.join(ROOTDIR, f'{source_model}_randnetw_indices.pkl'), 'wb') as f:
					pickle.dump(d_rand_idx, f)
			else: # load random indices
				d_rand_idx = pickle.load(open(os.path.join(ROOTDIR, f'{source_model}_randnetw_indices.pkl'), 'rb'))
			
			for k, v in state_dict.items():
				w = state_dict[k]
				# Load random indices
				print(f'________ Loading random indices from permuted architecture for {k} ________')
				idx = d_rand_idx[k]
				rand_w = w.view(-1)[idx].view(
					w.size())  # permute using the stored indices, and reshape back to original shape
				state_dict_rand[k] = rand_w
			
			enhance_model.modules['enhance_model'].load_state_dict(state_dict_rand)
			
		enhance_model.modules['enhance_model'].eval()
		
		### LOOP OVER AUDIO FILES ###
		for filename in tqdm(wav_files_paths):
			# filename = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds_16kHz/stim83_cicadas_TESTx10.wav'
			# Write hooks for the model
			save_output = SaveOutput(rand_netw=rand_netw)
			
			hook_handles = []
			layer_names = []
			layer_names_all = []
			for idx, layer in enumerate(enhance_model.modules['enhance_model'].modules()):
				# print(layer)
				layer_names_all.append(layer)
				if isinstance(layer, torch.nn.modules.Linear):
					handle = layer.register_forward_hook(save_output)
					hook_handles.append(handle)
					layer_names.append(layer)
				if isinstance(layer, torch.nn.modules.LSTM):
					handle = layer.register_forward_hook(save_output)
					hook_handles.append(handle)
					layer_names.append(layer)
			
			# Load and add fake batch dimension
			noisy = enhance_model.load_audio(join(DATADIR, filename)).unsqueeze(0)
			
			# Add relative length tensor
			enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
			
			## Detach activations
			detached_activations = save_output.detach_activations_metricGAN()
			
			# Get identifier (sound file name)
			id1 = filename.split('/')[-1]
			identifier = id1.split('.')[0]
			
			# Store and save activations
			save_output.store_activations(RESULTDIR=RESULTDIR, identifier=identifier)
			
			# Saving enhanced signal on disk
			torchaudio.save(
				f'pretrained_models/metricgan-plus-voicebank/outdir/enhanced_{identifier}_randnetw-{rand_netw}.wav',
				enhanced.cpu(), 16000)
	
	if source_model == 'ECAPA':
		classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa",
													savedir="pretrained_models/gurbansound8k_ecapa")
		out_prob, score, index, text_lab = classifier.classify_file('speechbrain/urbansound8k_ecapa/dog_bark.wav')
		print(text_lab)
	
	if source_model == 'transformerLM':
		asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
												   savedir="pretrained_models/asr-transformer-transformerlm-librispeech")
		asr_model.transcribe_file("speechbrain/asr-transformer-transformerlm-librispeech/example.wav")
	
	print('f')
