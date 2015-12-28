# CTC implementation for Blocks and Theano

Thomas Mesnard, Alex Auvolat

This repository contains an implementation of the CTC cost function (Graves et al., 2006). To avoid numerical underflow, two solutions are implemented:

- Normalization of the alphas at each timestep
- Calculations in the logarithmic domain

This repository also contains sample code for applying CTC to two datasets, a simple dummy dataset constituted of artificial data, and code to use the TIMIT dataset. The model on the TIMIT dataset is able to learn up to 50% phoneme accuracy using no handcrafted processing of the signal, but instead uses an end-to-end model composed of convolutions, LSTMs, and the CTC cost function. 

