# CcRNNs

Learning with recurrent neural network (RNN)
on long sequences is nontrivial due to the gradient vanishing or exploding issues. Recently, causal convolution has shown
promising performance in sequence modeling, and the convolutional architectures are found to be able to outperform
recurrent networks on tasks such as machine translation and natural language processing. Nevertheless, the network training
on extremely long sequences is still hard, and the main reason is that the efficient history length is quite limited. In order to
overcome the drawbacks of both RNNs and the convolutional architectures in sequence modeling, we propose an architecture
that can leverage both of their advantages. This is achieved by learning to embed causal convolution into the memory
breakpoints of a segmented-memory recurrent network. The causal convolutions are performed on the segments and used to
capture local dependencies in the forward pass. When integrated with an attention mechanism, the convolutions can provide long-
term information for the segmented-memory recurrent neural network. Note that in this architecture, the effective history field
of the causal convolutions does not need to cover the whole length of the sequential data, while it is essentially mandatory in
the conventional temporal convolutional network (TCN). Experimental results in diverse applications demonstrate superiority of
our method over several competitive counterparts including the temporal convolution network and the RNN variants. 

The Code of CcRNNs for training and testing on different datasets (tasks) are provided in the branches.
