# Variational Recurrent Neural Networks
This is an implementation of the paper "A Recurrent Latent Variable Model for Sequential Data".
http://arxiv.org/abs/1506.02216

Dependencies
------------
Most of the script files are written as pure Theano code, modules are implemented in a more general framework.
You can find the code at http://github.com/jych/cle.

Notice
------
The original Blizzard dataset should be downloaded by each user due to the license.<br>
http://www.synsig.org/index.php/Blizzard_Challenge_2013<br>
The original wave files have been read by numpy and saved into '.npz' format.
There is a function that reads the numpy formatted files and generate a hdf5 format file.
