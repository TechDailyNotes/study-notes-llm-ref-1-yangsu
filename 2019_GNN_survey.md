## Source

- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/pdf/1901.00596.pdf)

## Intro

### background

- directed acyclic graph
- recurrent GNN (RecGNNs)
    - represent node by propagating neighbor info iteratively until a stable point is reached
    - expensive
- convoluational GNN (ConvGNNs)
    - spectral methods
    - spatial methods
- graph autoencoders (GAEs)
    - network embedding
    - graph generation
- spatial-temporal GNN (STGNNs)

### GNN vs. Graph/Network Embedding

- GNN for various tasks with graph-like data
- network embedding for many methods targeting one task
- GNN can do network embedding by GAEs
- network embedding includes
    - GAEs
    - matrix factorization
    - random walks

### GNN vs. Graph Kernel Methods

- both embeds graphs to vector space by similarity measure and mapping
- GNN mapping is learnable, while not the case for graph kernel methods
- GNN definition
    - (normal) graph G = (V, E) with dim(V) = (n, d), dim(E) = (m, c)
    - directed graph -> adjacency matrix is asymmetric, converse to undirected graph
    - spatial-temporal graph G = (V, E, X_t) with dim(X_t) = (n, d), node attributes change over time

### Framework

- node-level
    - node classification & regression
- eg. semi-supervised learning for node-level classification
    - give a partial nodes labeled network
    - ConvGNNs learn a model that identify the labels for unlabeled nodes
- edge-level
    - edge classification and link prediction
    - give two nodes' hidden representation from GNNs as inputs
    - a NN can be trained to predict features of an edge
- graph-level
    - graph classification
    - obtain a compact representation of a graph by GNNs with *pooling* and *readout*
- eg. supervised learning for graph-level classification
    - graph conv layers perform node representations
    - graph pooling layers perform down-sampling
    - readout layer collapses all node representations into one graph representation
    - a MLP can be trained to predict graph labels from its representation
- eg. unsupervised learning for graph embedding
    - use autoencoder to learn node embeddings, want a decoder to be able to reconstruct the graph
    - use negative sampling to extract node pairs with no links, and positive samples are pairs with links, aim to distinguish them by logistic reg

## RecGNNs

- mostly pioneer works of GNNs
- mainly focus on DAG (directed acyclic graphs) due to compu. power
- RecGNN based on info diffusion mechanism
    - update nodes' states by exchanging neighbor info recurrently until a stable state is reached
    - to ensure convergence, recurrent function must be a *contraction mapping*
    - which shrinks the distance between two nodes after projecting them into a latent space
    - after convergence, a readout layer is used to forward propagates to a dummy super node
    - then a backward propagation technique is used to minimize the training objective
- GraphESN (Graph Echo State Network)
    - extended a echo state network on GNN
    - an encoder is built by a contractive state transition function
    - when node states recurrent updates are stable, a output layer is trained by taking the fixed states as inputs
- GGNN (Gated GNN)
    - use GRU (gated recurrent unit) as recurrent function
    - pros: GRU recurrent steps is fixed, no constraint is needed to ensure convergence
    - different from prev, use BPTT (back-propogation through time)
    - cons: BPTT run all nodes multiple times, expensive
- SSE (Stochastic Steady-state Embedding)
    - more scalable
    - sample batch of nodes for state update and gradient compu.
    - state update is done by weighted moving average of prev states (for stability)
    - but does not prove convergence

## ConvGNNs

- diff from RecGNNs, use a fixed number of weighted layers (or the number of recurrent steps is fixed)
    - spectral based: graph nodes are seen as graph signals, graph conv are removing noises from signals
    - spatial based: inherit from RecGNNs, graph conv are info propagation
    - GCN shows spatial includes spectral methods, so the latter is no longer used
- spectral ConvGNN (ignored)
    - see *graph Laplacian matrix*
    - see spectral CNN
    - see ChebNet
    - see CayleyNet
    - see GCN and extensions AGCN (Adaptive GCN), DGCN (dual GCN), and their PPMI (Positive pointwise mutual info) matrix techniques
- [spatial ConvGNN]
- NN4G (Neural Network for Graphs)
    - grap conv by summing node neighborhood info directly
    - the apply residual, skip connection to memorize node info
    - issue: use unnormalized adjacency matrix, causes node scales differ by a lot
    - later CGMM (Conextual Graph Markov Model) propose a probabilitic model using NN4G as backbone
- DCNN (Diffusion CNN)
    - treat graph conv as diffusion process
    - info transfer from one node to another with probability
    - and this info distribution can reach stability after some # rounds
    - define a % transition matrix *P* and applied on the input matrix *X* in each round
    - final output is a concatenation of the hidden representations of each round
- DGC (Diffusion Graph Conv)
    - learn from DCNN
    - uses summation instead of concatenation
    - because diffusion process is essentially summation of power series of P
    - since distant neighbors contributes very little, it implies P contributes very little in higher order power
- PGC-DGCNN
    - increase contribution of distant neighbors by shortest paths
    - define a shortest path adjacency matrix *S*
    - use CNN-like receptive field to do graph conv on S, P and X
    - also use concatenation like DCNN
    - issue: S is expensive to compute
- PGC (Partition Graph Conv)
    - partition node neighbors to Q groups by shortest path and something else (todo)
- MPNN (Message Passing NN)
    - a general framework for ConvGNNs
    - perform k-step message passing: namely spatial graph conv
    - learn a hidden representation in each step (h_k) by learnable functions M and U
    - for each node v: h_k = U_k(h_(k-1) on v, sum over neighbors u of v: M_k(h_(k-1) on v, h_(k-1) on u, input x on edge of u and v)); with the initial h_0 = x on v
    - node or graph level prediction task can be done by a readout function R on all h_k's
- [freq] GIN (Graph Isomorphism Network)
    - fix MPNN issue which cannot distinguish different graph structures
    - adjust weight of the central node by a learnabel paramater *e*
    - for each node v: h_k = MLP((1+e_k) * h_(k-1) on v + sum over neighbors u of v: h_(k-1) on u)
- [freq] GraphSage
    - fix issue that there may be too many neighbors of some nodes (which is expensive by previous approachs)
    - do sampling to obtain a fixed number of neighbors for each node
    - graph conv on each node v: h_k = scale s * (W_k * f_k(h_(k-1) on v, {h_(k-1) on u for all sampled u of v})), where f_k(.) is an aggregation function
- [freq] GAT (Graph Attention Network)
    - neighboring node weights are learnable by a NN (recall that in GCN, weights are fixed by the degree of u and v; in GraphSage, weights are the same)
    - use a_k on (v,u) to denote the connective strength (or weights) between node v and its neighbor u
    - a_k on (v,u) = softmax(LeakyRELU(a * [W_k * h_(k-1) on v | W_k * h_(k-1) on u])), softmax ensure weights sum to 1
    - graph conv on each node v: h_k = scale s * (sum over neighbors u of v: a_k on (v,u) * W_k * h_(k-1) on u), this is the multi-head attention step
- GAAN (Gated Attention Network)
    - GAT with an additional attention score for each attention head
- GeniePath
    - LSTM-like gating mechanism to control info flow across graph conv layers
- [freq] MoNet (Mixture Model Network)
    - a different approach to assign weights to (v,u)
    - introduce node pseudo-coordinates to determine relative position of (v,u), denote as p
    - once p is known, a weight function maps p to the weights of (v,u)
    - generalize many prev frameworks: GCNN (Geodesic CNN), ACNN (Anisotropic CNN), Spline CNN, GCN, DCNN by constructing nonparametric weight functions
    - additionally proposes a Gaussian kernel to learn weights
- PATCHY-SAN
    - rank a node's neighbors with a learnable weight
    - the ranking is based on graph structures only, expensive
- LGCN (large-scale GCN)
    - rank a node's neighbors by node feature info
    - build a feature matrix from neighborhoods and sort them
    - first q rows of the matrix are taken as input for the central node

### Improvement in terms of training efficiency

- base GCN training
    - very memory expensive as it is full batch, especially when there are millions of nodes
- GraphSage
    - use batch-training by sampling fixed k-step neighborhoods recursively
- FGCN (Fast Learning with GCN)
    - samples fixed number of nodes for each graph conv layer, instead of sampling fixed number of neighbors like GraphSage
    - interprets graph conv as integral transforms of embedding functions of nodes under probability measures
    - since node sampling are independent for each layer, between-layers connections are sparse
    - later another paper propose an adaptive layer-wise sampling approach and gets higher accuracy than FGCN at the cost of more complicated sampling scheme
- StoGCN (Stochastic Training of GCN)
    - reduce receptive field size of graph conv to an random small scale by conditioning on historical node representations
    - achieves good result even with only 2 neighbors
    - but needs to save intermediate state of all nodes, memory expensive
- Cluster-GCN
    - samples a subgraph by graph clustering alg, do graph conv on the subgraph instead
    - can handle very large graph with very small memory and same time as base GCN training

### Comparison between spectral and spatial models

- spectral model has a theoretical foundation in graph signal processing
- but spatial models are preferred due to efficiency, generality and flexibility issues
- 1. spectral model needs eigenvector compu. or handle the whole graph directly; but spatial model does conv directly in graph domain by info propagation, also can do batch-compu.
- 2. spectral model relys on graph Fourier basis, which generalizes poorly to new graphs (they assume a fixed graph), any perturbation to a graph result in change of eigenbasis; spatial models graph conv weights can be shared across different structures
- 3. spectral model is limited to undirected graphs only; spatial model can handle un/directed, signed, heterogeneous graph because they can be incorporated into aggregation function easily

### Graph Pooling Modules

- we want to use the learned GNN features to do all those downstream tasks
- but using all features are compu. hard, so down-sampling startegy is needed
- pooling
    - down-sampling nodes to generate smaller representations
    - also avoid overfitting, permutation invariance
- readout
    - generate graph-level representation by node-level representations
    - method is similar to pooling
- early work: ie. Graclus algorithm
    - use eigen-decomposition to coarsen graphs based on topological structure
    - time expensive
- recent: mean/max/sum pooling
    - h of graph at layer k = mean/max/sum of h_k of all nodes v
    - people have shown the importance of doing pooling at beginning of network to reduce dimensionality
- other works use attention to enhance mean/sum pooling
    - not efficient: fixed-size embedding is generated regardless of graph size
- Set2Set
    - build a memory embedding with size increase with input size
    - then build a LSTM to integrate order-dependent info into the memory embedding before pooling is applied
- ChebNet
    - rearranges graph nodes by Graclus alg and balanced binary tree
    - randomly aggregating this tree will arange similar nodes together
    - pooling on it is much more efficient
- DGCNN with SortPooling
    - node features from spatial graph conv are used to sort nodes
    - then truncates or extends node feature matrix to size of q
- [freq] DiffPool
    - prev approaches only consider graph features and ignore structural info of graphs
    - DiffPool generate hierarchical representations of graphs
    - learn a cluster assignment matrix *S* at layer k
    - S_k = softmax(ConvGNN_k(A_k, H_k))
    - the probability values in S are generated based on node features and topological structure, so that it learns both info of features and structures
    - issue: expensive as it generates dense graphs after this pooling
- SAGPool
    - DiffPool in a self-attention manner

### Discussion of Theoretical Aspects

- shape of receptive field
    - refers to the set of nodes that contributes to the final representation of the central node
    - all nodes of the graph would be included as layer k grows
    - thus ConvGNNs can extract global info by stacking graph conv layers
- VC dimension (few works)
    - results show that GNN model complexitiy increases very rapidlyz
- Graph isomorphism
    - two graphs are isomorphic if they are topological identical
    - if a GNN maps G1 and G2 to different embeddings, these 2 graphs are identified as non-isomorphic by the *WL test of isomorphism* (Weisfeiler-Lehman)
    - common GNNs like GCN and GraphSage cannot distinguish different graph structures
    - people also proved that GNN is at most as powerful as WL test if readout and aggregation functions are *injective*
- Equivariance and invariance
    - let f(A, X) be a GNN, Q be any permutation matrix that changes the order of nodes
    - GNN must be equivariant function when doing node-level tasks
    - GNN is equivariant if it satisfies f(QAQ^T, QX) = Q * f(A, X)
    - GNN must be invariant function when doing graph-level tasks
    - GNN is invariant if it satisfies f(QAQ^T, QX) = f(A, X)
- Universal approximation (very few works)
    - people proved that RecGNN can approximate any function that preserves *unfolding equivalence*
    - people proved that ConvGNN under message passing framework are not universal approximators of continuous functions defined on multisets (?)

## GAE (Graph Autoencoders)

### Network Embedding

- use an encoder to extract network embedding
- use a decoder to enforce the embedding to preserve graph topological info (ie. PPMI matrix, adjacency matrix)
- DNGR (deep NN for Graph Representations)
    - use stacked denoising autoencoder (MLP for both encoder and decoder) to reconstruct PPMI matrix by MLP
- SDNE (Structural Deep Network Embedding)
    - use stacked autoencoder (MLP for both encoder and decoder) to preserve the node 1st-order proximity and 2nd-order proximity
    - use two loss functions on encoder and decoder separately
    - loss function on encoder preserves node 1st-order proximity by minimizing the distance between node's embedding and its neighbor nodes' embedding
    - L_1 = sum over (u,v): A_vu * dist(enc(x_v) - enc(x_u)), A_v. = x_v
    - loss function on decoder preserves node 2nd-order proximity by minimizing the distance between node's input and its reconstructed input
    - L_2 = sum over v: dist((dec(enc(x_v)) - x_v) * b_v), b_v = 1 if A_uv = 0; b_v = beta > 1 if A_uv = 1
- GAE (Graph Autoencoder)
    - the former two approaches only uses node connectivity info, ignore node feature info
    - GAE reconstructs the adjacency matrix by using GCN to capture both
    - encoder Z = enc(X, A) = Gconv(ReLU(Gconv(A, X; theta_1)); theta_2), where theta are learnable parameters of GCN
    - Z now represents the network embedding matrix of the input graph with adjacency matrix A and node feature info matrix X
    - decoder A' = dec(z_v, z_u), which uses Z to reconstruct adjacency matrix A
    - the goal of decoder is to minimize the neg cross entropy loss between A and A'
- VGAE (Variational GAE)
    - deal with the problem of prev GAE that overfits by only reconstructing A
    - a variational GAE that learn the distribution of data
    - optimize the variational lower bound L = E_q(Z|X,A)[log p(A|Z)] - KL[q(Z|X,A) || p(Z)]
    - Kullback-Leibler divergence (KL) measures distance between two distributions
    - assumes the empirical distribution q(Z|X,A) is as close as possible to the prior distribution p(Z)
- ARVGA (Adversarially Regularized VGAE)
    - GAN + VGAE
- GraphSage
    - show that neg sampling loss can also enforce close nodes to have similar embedding and vice versa
    - L(z_v) = - log(dec(z_v, z_u)) - Q * E_(v' from P)[log(- dec(z_v, z_(v')))]
    - v' is a distant node of v, Q is the number of neg samples, P is the neg sample distribution
- DGI
    - capture global structural info by maximizing local mutual info
- DRNE (Deep Recursive Network Embedding)
    - assumes the aggregation of neighbor nodes' embedding should be similar to the central node's embedding
    - uses a LSTM to aggregate neighbors
    - the reconstruction error L = sum over v: dist(z_v - LSTM(z_u for u in a random sequence of node v's neighbors))
- NetRA (Network Representations with Adversarially Regularized Autoencoders)
    - propose a graph encoder-decoder framework with a general loss function
    - L = - E_(z from all data distribution P)[dist(z, dec(enc(z)))]
    - measures the expected distance between the real z and the reconstructed z
    - the encoder and decoder are LSTM with random walks rooted on each node v as inputs
    - both DRNE and NetRA regularizes the learned embeddings by prior distribution via adversarial training

### Graph Generation

- mainly used in drug discovery (molecular graph generation) by decoding graph structure given learned hidden representations
- has sequential and global methods
- sequential methods generate graphs by proposing node and edges one by one
- can grow until a certain criteria is satisfied
- DeepGMG (Deep Generative Model of Graphs)
    - assumes the probability of a graph is the sum over all node permutations
    - the node/edge adding decision is made by a updated RecGNN
- GraphRNN
    - two RNNs, one of the whole graph, one of the edges
    - the one on graph makes generation decision
    - the one on edge produces a sequence indicating the newly added node's relation with previous ones
- global approaches output graph at once
- GraphVAE
    - models existence of nodes and edges as indep rv. 
    - encoder is a posterior distribution q(z|G), uses ConvGNN
    - decoder is a generative Gaussian distribution p(G|z), uses MLP
- RGVAE (Regularized GraphVAE)
- MolGAN (Molecular GAN)
    - convGNN + GAN + RL
- NetGAN
    - LSTM + Wasserstein GAN
- conclusion
    - sequential methods linearize graphs into sequences, can lose structural info due to cycles
    - global methods are not scalable, O(n^2)

## Spatial-Temporal GNN (STGNN)

- tasks are mainly predicting spatial-temporal graph labels, forecasting future node values

### RNN-based approaches

- general
    - usual RNN: H_t = f(W * X_t + U * H_(t-1) + b), X_t is the node feature matrix at time t
    - add GCN: H_t = f(Gconv(X_t, A; W) + Gconv(H_(t-1), A; U) + b)
- GCRN (Graph Conv RNN)
    - LSTM + ChebNet
- DCRNN (Diffusion Conv RNN)
    - a diffusion graph conv layer -> GRU unit
    - it has an encoder-decoder framwork that can predict k step node values
- Structural-RNN
    - node-RNN + edge-RNN
    - spatial info is passed into them respectively
    - node-RNN takes output of edge-RNN to incorporate spatial info
    - also use semantic group of nodes and edges to save time
- issue
    - RNN propagation are expensive and have gradient explosion/vanishing issues

### CNN-based approaches

- general
    - uses parallel computing and have stable gradient, low memory benefits
    - has two CNNs
    - one 1D-CNN slides over the input matrix X along the time axis to aggregate temporal info
    - one Gconv aggregative spatial info of X at each time step
- CGCN
    - 1D-CNN + ChebNet/GCN
- ST-GCN
    - 1D-CNN + PGC layer (see PGC above)
- Graph WaveNet
    - instead of using pre-defined temporal graph, learn graphs automatically from data
    - propose a self-adaptive adjacency matrix A = Softmax(ReLU(E1 * E2^T))
    - E1, E2 denotes source and target node features
    - A learns the dependency between two nodes
- add attention
    - learn latent dynamic spatial dependency
    - GaAN: RNN-based + attention
    - ASTGCN: CNN-based + attention
    - issue: spatial dependency weight calculation is expensive, O(n^2)

### Conclusion

- datasets
    - citation networks: Cora, Citeseer, Pubmed, DBLP
    - biochemical graphs: NCI-1, NCI-9, MUTAG, D&D, PROTEIN, PTC, QM9, Alchemy, PPI
    - social networks: BlogCatalog, Reddit
    - others: MNIST, METR-LA, NELL
- open source implementations
    - pytorch geometric
    - dgl

### application

- inner tasks
    - node classification
    - graph classification
    - network embedding
    - graph generation
    - spatial-temporal graph forecasting
- graph-related tasks
    - node clustering
    - link prediction
    - graph partitioning
- cv-related tasks
    - scene graph generation (semantic relation recognition of ojects)
    - point clouds classification (by LiDAR scans)
    - action recognition (STGNNs)
    - human-object interaction
    - few-shot image classification
    - semantic segmentation
    - visual reasoning
    - question answering
- nlp-related tasks
    - text classification (syntactic dependency tree - syntactic relation among words, GCN)
    - neural machine translation
    - graph-to-sequence learning (abstract meaning representation, graph-LSTM, GGNN) -> knowledge discovery
    - sequence-to-graph learning
- traffic
    - traffic network forecasting (speed, volume, density by STGNNs)
    - taxi-demand prediction (LINE)
- recommender system
    - item, user recommendation
    - similar to link prediction
- chemistry
    - study graph structure of molecules/compounds
- others
    - program verification
    - program reasoning
    - social influence prediction
    - adversarial attacks prevention
    - electrical health records modeling
    - brain networks
    - event detection
    - combinatorial optimization

### future direction

- model depth
    - people show adding graph conv layers drop the performance of GNNs
    - question remained
- scalability trade-off
    - sampling may lose influential neighbor info
    - clustering may lose structural info
- heterogenity
    - no good methods for heterogenity graphs yet, most assume homogeneous
- dynamicity
    - need more temporal methods, now only STGNN