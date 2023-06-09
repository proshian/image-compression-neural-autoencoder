convert_weights.convert_weights takes src_w_path (to weights of depricated model) and dest_w_path where converted weights would be saved. The conversion needs so that new models can use the weights. The changes:

* normalising_activation is now a property of encoder. It used to be a property of NeuralImageCompressor.
* NeuralImageCompressor.encoder now should be of Encoder class.
* Encoder class consists of
    * backbone
    * feature_extraction
    * normalising_activation

This dir and all code inside will probably be deleted in a few commits