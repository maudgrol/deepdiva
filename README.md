# deepDIVA
DSR portfolio project for deep-learning syntheziser programmer using the vst plugin DIVA

###issues:
(1) preset_to_patch() breaks when i use a different preset than MS-REV1_deepdiva.h2p. (i put one HS-Brighton.h2p in data that breaks if I try to use it with preset_to_patch())

(2) our data generator can only generate batches of data, we need to fiddle those files together

(3*) the data generator just returns the list of tuples of the parameters we randomized. we could use my split_train_override_patch() together with preset_to_patch() to easily get the missing list of tuples of overridden parameters and add them (in case we need to do this at all... i would say we should input ONLY THE randomized parameters into the models anyway - all other parameters would not vary, therefore cannot be trained on.... )


###basic steps

[1] - transform between preset and patch , both ways (kind of done! but see issue 1)

[2] - generate data for a simple model 

[3] - setup a simple neural net and try to overfit it with the data

[4] - build the app that gets WAV file and returns preset

[5] - scale up steps [2] and [3] , try different features, experiment with model architectures...

[6] - make the app slicker and presentation preparation

[7] - set sail for the Autoencoder (we will probably get lost around here...)

