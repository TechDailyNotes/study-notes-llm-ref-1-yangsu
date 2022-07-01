# Graph Representation Learning

- [source: stanford CS224W - Machine Learning with Graphs 2021](http://web.stanford.edu/class/cs224w/index.html#content)

### Traditional Methods for ML on Graphs

**Node**

- Given G = (V, E), want to learn a function f: V -> R
- Node features
	- importance-based (important for predicting influential nodes in a graph): node degree, node Centrality
	- structure-based (important for predicting a particular role a node plays in a graph): node degree, clustering coefficient, graphlet count vector
- Node Centrality
	- *eigenvector centrality*: c_v = 1/lambda sum(neighbor node c_n of c_v) -> lamba c_v = A c_v, where A is the adjacency matrix, lamba is a constant, notice c_v is eigenvector of A
	- *betweenness centrality*: c_v = sum(all s!=t!=v: # shortest path between s and t that contains v / # shortest path between s and t)
	- *closeness centrality*: c_v = 1/sum(all u!=v: shortest path length between u and v)
- Clustering Coefficient
	- e_v = # edges among neighbor nodes / # max possible edges between neighbor nodes (k_v choose 2)
	- if e_v = 1, fully connected neighbors, e_v = 0, none of neighbors are connected with each other
	- observation: counts the # triangles in the *ego-network* (sub network induced by v and its neighbors)
	- so we can generalize the above by counting # pre-specified subgraphs, ie. graphlets
- Graphlets (rooted connected non-isomorphic subgraphs)
	- graphlet degree vector (GDV): count of # of graphlets rooted at node v -> measure a node's local topology

**Link**

- link prediction tasks
	- links missing at random
	- links over time
- distance-based features
	- shortest path distance between two nodes
- local neighborhood overlap
	- capture # neighboring nodes shared between two nodes
	- capture by counting
	- capture by *Jaccard's coefficient* (# common / # union)
	- capture by *Adamic-Adar index* (# common / degree of the neighbors)
- global neighborhood overlap
	- local -> 0 if two nodes have no neighbors in common
	- *Katz index*: count all # of paths between two nodes
	- use A (adjacency matrix), can prove P^k = A^k, where P^k is the # of paths between two nodes with length k
	- the actual Katz index adds a discount factor as path length increases: S = sum(beta^i * A^i), where beta is the discount factor

**Graph**

- graph kernels
	- key idea: bag-of-words for graph, represent a graph G by a vector f_G
	- for example: bag of node degree kernel
- graphlet kernel
	- count # different graphlets in a graph, here graphlet does not need to be rooted or connected
	- K(G, G') = h_G^T h_G' where h_G is the normalized version of f_G
	- h_G = f_G / sum(f_G), if we do not do this, if G and G' has different size, it will greatly skew the value of K(G, G')
	- limit: expensive
- *Weisfeiler-Lehman* kernel (color refinement)
	- generalized bag of node degree kernel
	- summarize k-hop neighborhood graph structure
	- efficient

### Node Embeddings

- (manual) feature engineering -> (auto-learnable) feature representation/embedding
- goal: similarity in the original network -> similarity of the embedding
	- node u, v -[encoder]-> embeddings z_u, z_v (d-dim)
	- define a node similarity function s
	- embeddings (u, v) -[decoder]-> similarity score s(u, v)
	- optimize encoder param s.t. s(u, v) = z_v^T z_u
- simplest: embedding-lookup
	- z_u = Z * one_hot(u), where dim(Z) = d * |V| is what we want to learn
	- very expensive, but network structure is preserved
	- each node is assigned a unique embedding vector
	- i.e. DeepWalk, node2vec (differ by similarity function)
	- note that this is unsupervised/self-supervised
	- and the embeddings are task dependent

**Random Walk**

- s(u, v): % that u and v co-occur on the random walk
- expressivity: random walk is a *flexible stochastic* definition of node similarity, incorporates both local and higher-order neighborhood info
- efficiency: do not need to consider every node pair, only those co-occur
- alg
	- goal: learn a mapping f(u) = z_u
	- define N_R(u) to be the neighborhood of node u by random walk strategy R
	- objective: max f: sum(u, log P(N_R(u) | z_u))

**DeepWalk**

- optim
	- run *short fixed-length, unbiased* random walk for each u using R
	- for each node u collect N_R(u), the multiset of nodes visited from u
	- optimize the objective
	- loss: sum(u in V, sum(v in N_R(u), -log P(v | z_u)))
	- where P(v | z_u) = exp(z_u^T z_v) / sum(n in V, exp(z_u^T z_n))
- intuition of the loss
	- the outer is sum over all nodes u in the graph
	- the inner is sum over nodes v seen on random walks starting from u
	- P is the % of u, v co-occuring on the random walk
	- finally we minimize the loss

**Node2Vec**

- optim
	- note the runtime is |V|^2 from the outer sum and the denominator of P (the softmax), expensive
	- do *negative sampling*: normalize against k negative samples instead of all nodes
	- sample k nodes with % wrt. their degree
	- -log P(v | z_u) ~= log sigmoid(z_u^T z_v) - sum(i=1->k, log sigmoid(z_u^T z_ni)), where ni ~ P_V (random dist over nodes)
- *2nd order biased* random walk
	- use flexible, biased walks that do trade off between *local and global* view of the network
	- i.e. BFS (micro-view), DFS (macro-view)
	- define 2 param
	- return param p: return back to prev node
	- walk away param q: moving outwards %, essentially the ratio of BFS & DFS
- alg is parallelizable and with linear complexity
	- compute random walk %
	- simulate r random walks of length l from each node u
	- optimize objective (SGD)
	- drawback: need re-learn with bigger network

**Embedding entire graphs**

- simplest idea
	- run usual node embedding
	- sum/mean over all embedding -> graph embedding
- virtual node idea
	- add a virtual super node to represent the graph
	- it can connect to all nodes for example
- anonymous walk embeddings
	- construct a list corresponds to the index of the first time each node is visited in a random walk
	- simulate walks of l steps and record their counts
	- represent graph as % distribution over these walks
	- i.e. if l=3, graph embedding is then a 5-dim vector (since there are 5 anonymous walk of length 3)
	- how many anonymous walk to sample: m = ... (has a proof)
- enhance: learn walk embeddings
	- learn a embedding zi of walk wi
	- then learn a graph embedding zG *together* with all the walk embeddings zi (so treat zG as a walk as well)
	- the idea is to embed walks s.t. the next walk can be predicted
	- sample T walks, want to predict walks that co-occur in a d-size window
	- objective: max zG: sum(t=d+1->T, log P(w_t | w_t-d, ..., w_t-1, zG))
- next: hierarchical embeddings
	- graph has their local community (subgraphs)
	- GNN

### PageRank

- web as directed graph -> rank nodes by *link analysis algorithms*
	- pagerank
	- personalized pagerank (PPR)
	- random walk with restarts

**PageRank**

- alg
	- the rank for node j is the sum of the split of in-link ranks
	- rj = sum(i->j, ri/di), where di is the out-degree of node i
	- expensive
- matrix formulation
	- stochastic adjacency matrix M
	- Mij = 1/dj if j -> i
	- let ri be the importance score of page i
	- sum(i, ri) = 1
	- now r = M * r
- stationary distribution
	- if the rank of node j at time t is p(t)
	- then at time t+1, p(t+1) = M * p(t)
	- if p(t+1) lim= M * p(t) = p(t), p(t) is stationary
	- hence reaches r = M * r
- eigenvector formulation
	- r can be viewed as both the principle eigenvector of M
	- and as the stationary distribution of a random walk over the graph
	- but it is expensive to calculate, thus need a iterative alg
- solve pagerank
	- assign each node an initial rank
	- repeat p(t+1) = M * p(t) until the diff of p(t+1) and p(t) is less than some threshold sigma
	- power iteration: initialize equally, l1 diff -> stop about 50 iterations
- spider trap problem
	- node has self-loop and no out-link
	- all importance gets absorbed
	- solve: add prob (1-beta) to jump out randomly, (beta) to do normal stuff
- dead end problem
	- node has no out-link, no self-loop
	- solve: uniformly jump out randomly
- combined pagerank
	- rj = sum(i->j, beta * ri/di) + (1-beta) * 1/N
	- matrix form
	- P = beta * M + (1-beta) * [1/N]\_(NxN)

**Random Walk with Restart**

- node proximity measurements
	- shortest path
	- common neighbors
- personalized (topic-specific) pagerank
	- ranks node proximity to teleport nodes S
	- if there is only one S (we care about), always teleport to S -> random walk with restart (S is called the starting node)
- alg ex. in user-item bipartite graph
	- sample an item from query nodes S (could sample by weight)
	- randomly choose a neighbor user of the item
	- randomly choose an neighbor item of the user
	- item count increase
	- if % choose to restart, repeat
- benefits (utilizes..)
	- multiple connections, multiple paths
	- direct and indirect connections
	- degree of nodes

**Matrix Factorization vs. Node Embedding**

- simplest node similarity: they are connected by edges
	- i.e. z_v^T z_u = A_(u,v) where z is the node embedding, A is the adjacency matrix
	- so we want A to be factorized into Z^T Z
	- the exact solution is generally not possible
	- but can learn Z by minimizing ||A - Z^T Z||\_2
- random walk based similarity
	- can also be formulated, but more complex
- limitation of both methods
	- cannot obtain embedding for newly added nodes
	- cannot capture local structural similarity (similar subgraphs)
	- cannot utilize the features (of nodes or edges or whole graph)
	- solution: GNN

### Message Passing

- *homophily*: the tendency of individuals to associate and bond with similar others, i.e. researchers in the same areas are more likely to know each other
- *influence*: social connections can infulence the individual characteristics of a person
- motivation
	- guilt-by-association: if i am connected to a node with label X, i am likely to be labeled X as well
	- classification label of a node may depend on: its features, label and features of its neighborhood
- (first order) *markov assumption*
	- label Y_v of node v only depends on its neighbors N_v: P(Y_v) = P(Y_v | N_v)
	- local classifier: used for initial label assignment, label nodes based on its features only (no network info is used)
	- relational classifier: label based on neighbors info (label, attributes)
	- collective inference: propagate the correlation, apply relational classifier iteratively until convergence
- (probabilistic) relational classifier
	- class probability Y_v of node v is avg class probability of its neighbors
	- for labeled nodes v, initialize them with ground truth
	- for unlabeled nodes, initialize them with % = 0.5
	- update all nodes in random order until convergence or some max # iterations
	- update: % that Y_v is class c = avg of the neighbors % in class c
	- (note that this does not use node features)
- iterative classification
	- class probability Y_v of node v depends on both the *summary* z_v of its neighbors N_v 's class and its features
	- z_v could be: histogram of the # of each label in N_v, most common label in N_v, # of different labels in N_v
	- now given a partially labeled network, set labeled as train set and unlabeled as test set
	- train a classifier F1 based on node features Y_v on train set
	- also train a classifier F2 based on both node features Y_v and summary vector z_v of node neighbors on train set
	- on test set, do the following until convergence of some # of iterations
		- 1. predict Y_v by F1
		- 2. update z_v based on new prediction
		- 3. reclassify all nodes by F2
		- 4. update Y_v by F2 based on new labels
		- repeat step 2
- (loopy) belief propagation
	- a DP approach to answer probability queries in a network
	- each nodes listens to the message from its neighbors, updates it and passes it forward
	- notation
		- label-label potential matrix P: the % that i am in class Y_i given my neighbor j is in class Y_j
		- prior belief B: % that i am in class Y_i
		- m_(i->j, Y_j) to denote the message from i that the % j is in class Y_j
		- L: set of all labels
	- then for node j, m_(i->j, Y_j) = sum over all label i in L: (the label-label potential P_ij * the prior B_i * product over all neighbors k of j: m_(k->i, Y_i) which contains all messages sent by neighbors of i from the previous round)
	- after convergence, the belief of node i in class Y_i = B_i * product over all neighbors j of i: m_(j->i, Y_i)
	- cons: if network has cycles, beliefs may not converge (like spider trap in PageRank), because messages may be dependent on each other
	- pros: easy to parallelize, can apply to any graph model

### GNN 1: The Model

- naive approach
	- use adjacency matrix A to train a NN
	- expensive
	- overfit as param = |V|^2, V = # node = # training examples
	- not invariant to node ordering
- idea from CNN
	- define a notion of locality/sliding window on the graph
	- network neighborhood defines a computation graph
	- every node has a comp graph (NN) based on its neighborhood
	- node embedding is generated by weighted aggregating local information
	- layer-k embedding gets info from nodes k-hop away
- neighborhood aggregation
	- need to be order/permutation invariant, as node orderings are
	- i.e. avg, sum
	- initial layer-0 embedding h_v = node feature x_v
	- layer-(l+1) embedding h_v = non-linear trans: (W_l * avg or sum over neighbors u: layer l embedding h_u + B_l * layer-l h_v
	- embedding after L layers neighborhood aggregation z_v = layer-L h_v
- matrix formulation
	- layer-(l+1) embedding matrix H = non-linear trans: (A' * layer-l embedding matrix H * W_l^T + layer-l embedding matrix H * B_l^T)
	- where A' = D^(-1) * A, D is a diagonal matrix that D_(v,v) = Deg(v), so it is like avg over the adjacency matrix A
	- here A'HW_l^T refers to neighborhood aggregation
	- and HB_l^T is self-transformation
- how to train it
	- unsupervised: use graph structure, i.e. node similarity, matrix factorization, node proximity, etc.
	- supervised: i.e. CE loss in node classification
- inductive node embeddings
	- learned W,B can generalize to unseen nodes, graphs

### GNN 2: The Design Space

- general GNN design space
	- message passing (how to pass info)
	- aggregation (how to combine info) + activation
	- layer connectivity (how to stack multiple layers)
	- graph manipulation (how to build computation graph)
	- learning objective (supervised/unsupervised, node/edge/graph level)
- GCN
	- message + aggregation is simply the mean of neighbor messages * W -> af
	- 'af' refers to any activation function: ReLU, Sigmoid, Parameteric ReLU
- GraphSAGE
	- GCN + arbitrary aggregation function: mean, pool, lstm, etc.
	- after aggregating neighbors, concat with myself at prev layer
	- so h_v = concat(agg(m_u, u in neighbor of v), m_v) * W -> af
	- can also apply l2 normalization before activation
- GAT
	- add an attention weight alpha to each edge before aggregation
	- let attention coefficient e_uv = A(W' * h_v, W' * h_u), where A is some transformation function, W' is a different attention matrix at each layer
	- alpha = e_uv normalized by softmax so that all neighbors of v sum to 1
	- A(.,.) can be a simple linear layer of concat(.,.)
	- multi-head attention: can have many A, then agg all result
	- so h_v = agg(h_v[i] for i in # heads), where h_v[i] = agg(alpha[i] * h_u) * W -> af
	- key benefit: implicitly specifying different alpha (importance values) to different neighbors
- stacking GNN layers
	- standard way: stack sequentially
	- over-smoothing problem: the issue of stacking many GNN layers
	- lesson 1: do not use too many layers
	- solution 1: add multiple MLP/transformations within a single GNN layer
	- solution 2: add layers that do not pass messages (MLP->...->MLP->GNN layer->...->GNN layer->MLP->...->MLP)
	- lesson 2: add skip connections in GNNs (h_v = F(.) + h_v of prev layer -> af, where F(.) is the original GNN agg function)

### GNN Application

**GNN Augmentation** (design space - layer connectivity)

- assumption: the raw input graph == computational graph
- break the assumption - features: input graph lack features
	- solution: feature augmentation
- break the assumption - graph structure: input graph ...
	- too sparse -> inefficient message passing
		- solution: add virtual nodes/edges
	- too dense -> message passing too costly
		- solution: sample neighbors when doing message passing
	- too large -> cannot fit whole computational graph into memory
		- solution: sample subgraphs to compute embeddings
- feature augmentation
	- if input graph has no node feature
		- assign unique IDs to ndoes (ID -> one-hot vector)
		- issue: cannot work on new nodes
		- assign constant node features (ie. 1 for every node)
		- issue: can only learn structural info
	- why need
		- certain structure is hard to learn by GNN (ie. cycle count feat)
- add virtual edges
	- common: connect 2-hop neighbors via virtual edges (ie. bipartitie graph)
	- intuition: use A + A^2 instead of A (adj matrix) for GNN computation
- add virtual nodes
	- a virtual node connect to all nodes in the graph
	- then every nodes has a distance of 2
	- benefit: greatly improves message passing in sparse graph
- sample neigbors
	- greatly reduce compu. cost, scale to large graphs

**GNN Training** (design space - learning objective)

- prediction heads
	- node-level: directly use node embeddings
		- k-way classify: class = h_v * W, where dim(W) = (., k)
	- edge-level
		- k-way classify: class = Head(h_v, h_u)
		- Head(.) can be concat + linear
		- link prediction -> like 1-way classify: can use dot product Head(.)
		- also can use multi-head attention for k-way classify: class_k = h_v^T * W_k * h_u, then class = concat(class_i for i in k)
	- graph-level
		- Head(all h_v)
		- global pooling: Head(.) = mean/max/sum, not good for large graph (ok for small graph)
		- hierarchical global pooling: ie. y1 = Head(some nodes), y2 = Head(other nodes), final y = Head(y1, y2)
		- DiffPool: hierarchically pool node embeddings, use 2 GNNs: one is standard, the other compute the cluster each node belongs to -> can be trained in parallel
- supervised vs. un(self-)supervised
- loss function (classify/regression/ranking)
	- cross-entropy/MSE/max margin
- evaluation metric(k-class classify/binary classify/regression)
	- multi-class accuracy/ROC AUC/RMSE
- split data to train/valid/test
	- in graph, data leakage is possible caz nodes are connected
	- solution 1: transductive setting - only split node labels on the entire graph
	- solution 2: inductive setting - break the edges between splits to get multiple (3) graphs
	- in link prediction, data split is tricky
		- assign two types of edges in the original graph
		- message edge: use for message passing
		- supervision edge: use for computing objective
		- inductive: 3 graphs each with different supervision edges
		- transductive (default): nested graph, train < valid < test (full graph) -> as if a dynamic graph evolving through time

### Theory of GNN

- how well can GNN distinguish different graph structure, if all node features are the same
	- consider local neighborhood structures: cannot distinguish symmetric (isomorphic) nodes
	- GNN generate same embedding for nodes with same computational graph (i.e. rooted subtree structure)
- most expressive GNNs should always map subtree with different structure to different embeddings
	- each step of aggregation should fully retain neighboring info
	- aka. the neighbor aggregation is *injective* (no different input would yields same output)
	- this way no info gets lost
- analyze the expressive power of GNNs
	- GCN w/ mean pooling (fail)
	- GraphSAGE w/ max pooling (fail)
- GIN: the most powerful GNN
	- GIN is an injective function of the form of F(sum(a set of x, F(x))) that does not lose information
	- GIN is a NN version of *WL graph kernel*
- WL graph kernel (color refinement algorithm)
	- assign a color c_0(v) to each node v
	- at each layer, the next layer of c(v) = Hash(c(v), neighbors of c(v))
	- then after k steps, c(v) summaries the structure of k-hop neighborhoods
	- two graphs are isomorphic if their color refinement is the same (ie. indistinguishable)
	- which can be modeled as ...
	- at each layer, GINConv(c(v), all neighbors c(u)) = MLP_1((1+l) * c(v) + sum(all c(u))) where l is a learnable param
	- so GIN is a lower-dim embedding of WL graph kernel (which is one-hot)
- summary
	- the most powerful GNN is an *injective multi-set function*
	- the key is to use element-wise sum pooling, instead of mean/max pooling
- how to further improve
	- if node & edge features are different

### Knowledge Graph Embeddings

- heterogeneous graph
	- biomedical KG
	- event graph
- RGCN (relational GCN)
	- GCN on different edge relation, with self-loop and normalized by each relation
	- issue: big # param if too many relations
	- solution 1: block diagonal matrices - make weight matrix sparse
	- solution 2: basis learning - share weights across relations
		- matrix of each relation is a linear comb of some # (b) of basis transformation matrices
		- (b) is self-defined, can be much smaller than # relations if the latter is big
- downstream task
	- node classification (same)
	- link prediction
		- need to split edges to train/test independently by relation type to avoid rare edge type gets split to 0
		- can also create negative edges (corrupt edge type to be another) when training
		- objective is then to rank supervision edges score over negative edges
- extension
	- RGraphSAGE
	- RGAT

**KG Completion**

- concept
	- KG is heterogeneous graph that captures factual info in some domain
	- KG Completion predict missing tails t given head h and relation r
	- ie. embedding of (h applied on r) should be close to embedding of t
- relation patterns
	- (anti)symmetric relation: (h,r,t) -> (not) (t,r,h)
	- inverse relation: (h,r1,t) -> (t,r2,h)
	- composition (transitive) relation: (x,r1,y), (y,r2,z) -> (x,r3,z)
	- 1-to-N relation: (h,r,tx) for x in (1,...,N) are all true
- TransE
	- want h + r = t if (h,r,t) triplet holds
	- contrastive loss: favors low distance for valid triplets, and favors high distance for corrputed ones
	- want to minimize sum(batch: margin + d(h + r, t) - d(h' + r, t')) where (h,t) are positive pairs, (h',t') are negative (sampled) pairs
	- what relations can TransE capture
		- antisymmetric: h + r = t -> t + r != h
		- inverse: h + r1 = t -> t + r2 = h if r2 = -r1
		- composition: x + r1 = y, y + r2 = z -> x + r3 = z if r3 = r1 + r2
	- what relations cannot TransE capture
		- symmetric
		- 1-to-N because tx for x in (1,...,N) would all be mapped same, even if they are different entities
- TransR
	- TransE translates any relation in the *same* emebdding space
	- TransR creates a new embedding space for relation
	- want Mr * h + r = Mr * t, Mr is the projection matrix
	- what relations can TransR capture
		- symmetric by setting r = 0: Mr * h = Mr * t -> Mr * t = Mr * h
		- antisymmetric: Mr * h + r = Mr * t -> Mr * t + r != Mr * h
		- inverse: same r2 = -r1
		- 1-to-N: now Mr * t1 = Mr * t2 = ..., where t1 != t2 is possible
	- what relations cannot TransR capture
		- composition: Mr1 * x + r1 = Mr1 * y, Mr2 * y + r2 = Mr2 * z -> Mr3 * x + r3 = Mr3 * z is hard, because Mr3, r3 is high dimensional and is not *naturally compositional*
- Bilinar Modeling: use another distance measurement, ie. distmult
- DistMult
	- measure (h,r,t) by sum(i: hi * ri * ti) for i in dim(h)
	- can be viewed as cosine similarity between h * r and t
	- what relations can DistMult capture
		- symmetric: naturally symmetric because DistMult is communitive
		- 1-to-N: tx can be on the same hyperplane
	- what relations cannot DistMult capture
		- antisymmetric
		- inverse: cannot, because the only way this works is to make r1 and r2 the same (ie. symmetric)
		- composition: cannot express (r1,r2) by a single hyperplane (ie. r3)
- ComplEx
	- DistMult but embed into complex-value instead of real-value
	- score function is Re(sum(i: hi * ri * conj(ti)))
	- what relations can ComplEx capture
		- symmetric by setting Im(r) = 0: hi * conj(ti) = conj(ti) * ti
		- antisymmetric: by hi = conj(ti) and conj(hi) = ti
		- inverse: r1 = conj(r2)
		- 1-to-N: like DistMult
	- what relations cannot ComplEx capture
		- composition: like DistMult
- conclusion
	- first try TransE to quickly see if KG does not have much symmetric relations
	- then use more expressive models as above

### Reasoning over KG

- multi-hop reasoning (ie. answer complex queries on incomplete big KG)
	- one-hop queries: KG Completion, find t that satisfies (h,r,t)
	- path queries: find all rx satisfy (h,[r1,...,rn],t)
	- conjunctive queries: find hx satify (hx,rx,t) given multiple rx
- intuition
	- note that (h,r,t) may be missing, so we cannot do dictionary-searching (existing) path
	- also, we cannot check if every (h,r,t) is possible because the complete graph is too dense (high complexity)
	- we want to do "predictive queries", ie. answer all queries given an incomplete graph
- embed queries (one-hop & path queries)
	- generalize TransE to multi-hop reasoning
	- minimize |q - t|, where q = h + r1 + ... + rn
	- firstly train a KG by method X, then path queries can be answered by traversing KG embedding as long as *composition* relation rule is satisified by KG method X
	- can also use trained KG embedding to impute missing nodes
- embed queries (conjunctive queries)
	- embed queries with hyper-rectangles (boxes)
	- q = (center(q), offset(q))
	- query-to-box

**Query-to-Box**

- use box because the *intersection* of boxes is well-defined
	- easy to calculate
	- the intersection is still a box
- input
	- entity embeddings, relation embeddings, intersection operator f(box x ... x box -> box)
	- f: want the output box center be close to all input box centers
	- also the output size should be shrinked (because of intersection)
- algorithm
	- center(q_output) = sum(wi * center(qi_input)), where wi = softmax_j(F_cen(center(qi))), and F_cen is learnable network
	- offset(q_output) = min(offset(qi_input)) * sigmoid(F_off(qi_input)), and F_off is also learnable
- define entity-to-box distance (AND)
	- given a query box q and entity box v, dist(q,v) = d_out(q,v) + a * d_in(q,v)
	- where d_out is dist function outside of the box, d_in similarly with a penalty 0 < a < 1
	- so if point is in the box, distance is downweighted
- further: union conjunctive
	- conjunctive queries + disjunction = Existential Positive First-order (EPFO) queries, we refer to them as AND-OR queries
	- challenge: union of some queries may overlap other query boxes in low-dim vector space
	- conclusion: given M conjunctive queries with non-overlapping answer boxes, we need O(M) to handle all OR queries
	- this is too expensive
	- better if using AND-OR: push all union to the last step after doing AND, since intersection decreases boxes size to the minimal, this is called disjunctive normal form (DNF)
- define entity-to-box distance (OR)
	- d_box(q,v) = min(i=0->m, d_box(qi,v))
- training
	- randomly samples q from training graph, answer v and a neg sample v'
	- embed query q
	- calculate score f_q(v) and f_q(v'), ie. the neg distance function
	- optimize the loss l to maximize f_q(v) and minimize f_q(v')
	- l = - log(f_q(v)) - log(1 - f_q(v'))
- how to generate a complex query q
	- start with a *query template*
	- generate query by instantiating every variable with a concrete entity and relation from the KG
- how to instantiate query template given KG
	- randomly instantiate a root node of the query template
	- randomly sample the projection edge with the relation associated with our selected root
	- the node connected by the edge becomes the anchor node, repeat the process until q is complete
	- note that q must have answers on the KG and one of the answers is the instantiated root
- can visualize the embedding by t-SNE
- summary: the key is to embed queries by navigating the embedding space

### Frequrent Subgraph Mining with GNN

### GNN in Recommender System

### Community Structure in GNN

### Deep Generative Models for GNN

### Advanced Topics of GNN

- a "perfect" GNN
	- an injective function: neighbor structure -> different node embeddings
	- observation 1: if two nodes have same neighborhoods, they have the same embedding
	- issue: we may want they have diff embeddings even their neighbors are the same, because the nodes have diff positions in the graph
	- solution: position-aware GNNs
	- observation 2: if two nodes have diff neighborhoods, they have diff embeddings
	- challenge: this is hard to achieve with no useful node features, bounded by the WL test
	- solution: identity-aware GNNs
- naive solution
	- assign one-hot encoding to every node
	- not scalable, feature dimension scales with # nodes
	- not inductive, cannot generalize to new nodes/graphs

**Position-Aware GNN**

- two tasks
	- struture-aware: nodes are labeled by their neighborhood strutures
		- GNNs often work well by differentiating computation graph
	- position-aware: nodes are labeled by their relative positions in the graph
		- GNNs always fail due to structure symmetry
- "Anchor": reference point => positional encoding
	- randomly pick a node s1 as anchor node
	- represent v1, v2 by their relative distance to s1, which is different
	- anchor node => coordinate axis to locate nodes
	- pick more nodes s1, s2 as anchor nodes
	- many anchor nodes => many coordinate axes
	- pick a connected set of nodes as anchor sets
	- obs: large anchor sets can provide more precise position estimate
- how to use position info
	- simple: use as an augmented node feature
	- issue: since each dim of positional encoding is tied to a random anchor, dims of positional encoding can be randomly permuted without changing its meaning
	- but in a NN, randomly permuting dim of the input will change the output
	- solution: a special NN to maintain permutation invariant property of position encoding (ie. sum aggregator -> read paper for details)

**Identity-Aware GNN**

- how normal GNN fails in structural task
	- node-level: diff input with same computation graph
	- edge-level: in link prediction, nodes have the same embedding because of the same computation graph
	- graph-level: diff graph with same computation graph for all nodes (complex but could happen)
- inductive node coloring
	- idea: assign a color (unique label) to the node we want to embed
	- inductive because the coloring is invariant to node ordering -> generalize better
	- when computation graph is deep, it will cycle back to the starting node, with the unique color we assigned
- how ID GNN succeeds in structural task
	- node-level: color root nodes with identity -> diff computation graph
	- edge-level: in link prediction, we can classify (v0, v1), (v0, v2) by coloring v0 to force v1 and v2 have diff colored computation graph
	- graph-level: use two types of nodes, ones with augmented color identity, ones without -> diff computation graph
- how to build an ID-GNN
	- different message passing/aggregation for diff-colored nodes -> heterogeneous message passing
	- intuition: ID-GNN is actually counting cycles with diff-lengths originating from a given node
	- simplified ID-GNN: use cycle counts in each layer as an augmented node feature
- ID-GNN is more expressive than 1-WL test

**Robust GNN**

- attack possibilities
	- target node: node prediction we want to change
	- attacker node: node the attack can modify
	- direct attack: attacker node == target node
		- modify target node feature
		- add/remove connection to/from the target node
	- indirect attack: attacker node != target node
		- modify attacker node feature
		- add/remove connection to/from the attacker node
- the GNN model that we are attacking
- mathematical formulation
	- objective: maximize (change of target node label prediction) subject to (graph manipulation is small)
	- original graph: A adjacency matrix, X feature matrix
	- manipulated graph aftering adding noise: A', X'
	- assumption: (A', X') close to (A, X)
	- original param: P = argmin(P: Loss on Train(P; A, X))
	- orignal prediction: C_v = argmax(C: P(A, X)) given C and v where C_v refers to the class of vertex v that has highest predicted %
	- manipulated param: P' = argmin(P': Loss on Train(A', X'))
	- manipulated prediction: C_v' = argmax(C': P'(A',X')) given C' and v
	- want C_v != C_v'
	- want the change of prediction on target node v to be maximized, ie. Gradient(v; A', X') = log(P'(A',X')) given C' and v - log(P'(A',X')) given C and v, subject to (A',X') \~= (A,X)
	- the 1st term is the predicted log % of the newly-predicted class C' on v, we want to increase this
	- the 2nd term is the predicted log % of the originally-predicted class C on v, we want to decrease this
- challenge
	- A' is a discrete object, not weight that can be gradient-descented
	- for every modified (A',X'), GNN needs to be retrained -> expensive
- empirical result of attack comparison
	- direct attack >> indirect attack \~= random attack > no attack

### Scalable GNN

- challenge of minibatch training on GNN
	- randomly sample nodes does not work
		- no message passing
	- naive full batch training
		- all nodes embedding at current layer -> next layer
		- not feasible for large graph (too much memory for GPU)
- Neighbor Sampling (proposed in GraphSAGE)
	- sample only on k-hop neighborhood subgraph
	- M node batch size -> M computation graphs
	- issue: full neighbor computation graph is still big if we reach *hub node* (many neighbors)
	- solution: use fixed H # of neighbor sampling
	- time complexity: M * H^k if k layers
	- trade-off in sampling H #: efficient but larger variance/unstable
	- computation time: computation graph size still exponential wrt. # layers
	- sample randomly: may sample many unimportant nodes
	- random walk with restarts: sample with scores (importance)
- Cluster-GCN
	- computation graph of many nodes have same substructure (redundancy)
	- "vanilla" Cluster-GCN: partition subgraphs then do full-batch training on each subgraph
	- which subgraphs are good: retain the community structure, have more connectivity pattern (less isolated nodes)
	- can partition using existing community dectection algorithm, ie. SETIS
	- issue 1: the induced subgraphs remove between-group links -> hurt GNN performance
	- issue 2: sampled nodes are not diverse enough to represent the graph structure, each subgraph is concentrated on their aspects so the training is unstable
	- solution: partition graph into smaller groups of nodes, then do mini-batch training on multiple node groups
	- "advanced" CLuster-GCN: partition more subgraphs then do mini-batch training on multiple subgraphs, with the edges between the batched subgraphs connected
	- time complexity: k * M * D_avg if k layers, M nodes, if D_avg = 2H, this is much more efficient than neighbor sampling, especially when k is large
	- still leads to systematically biased gradient estimate due to missing cross-community edges
- Simplified GCN
	- simplify by removing the non-linear activation
	- original hi = ReLU(normalized(sum(all j: hj)) * W_(k-1)) at layer k
	- in matrix form: H_(k) = ReLU(A' * H_(k-1) * W_(k-1)^T), where A' is the normalized version of A
	- without activation, H_k = A' * H_(k-1) * W_(k-1)^T = A'(A' * H_(k-2) * W_(k-2)^T) * W_(k-1)^T = ... = A'^k * H_0 * W^T where W^T = W_(k-1) * W_(k-2) * ... * W_(0)
	- note A'^(k) * H_0 can be pre-computed
	- so H_k = X * W^T where X = A'^(k) * H_0
	- for every node, hi = W * Xi at layer k
	- so we directly get all nodes' embeddings at layer k in linear time
	- for mini-batch of size n, SGD on n node embeddings -> n W matrix
	- pros: compared to neighbor sampling: much more efficient
	- pros: compared to Cluster-GCN: nodes can be sampled randomly
	- cons: far less expressive because of the lack of non-linearity
	- but in real world it does not hurt that much
	- reason: *graph homophily*
		- nodes connected by edges tend to share the same features/labels
		- the pre-processed X is obtained by iteratively averaging neighboring nodes' features
		- so simplified GCN aligns well with graph homophily
		- as a result, simplified GCN works very well in node classification
		- in short it is good if the graph is simple

### Other Topics of GNN

**GNN in Computational Biology**

**Pre-training GNN**

**Hyperbolic Graph Embeddings**

**Design Space of GNN**

- intra-layer design: GNN layer = transformation + aggregation
- inter-layer design
	- pre-process layer: ie. initial MLP/embedding transformation, important when expressive node feature encoder is needed -> text/image as nodes
	- skip-connection: improve deep GNN's performance
	- post-process layer: ie. final MLP, important when reasoning or transformation over node embeddings are needed -> graph classification, KG
- learning configuration: lr, batch_size, layers, etc.
- quantitative task similarity metric
	- select *anchor* models M1, ..., Mn by randomly sample N models from the design space (ie. all possible param combinations) and evenly select n from N based on their sorted performance -> goal: cover a wide range of models
	- rank the performance of anchor models on T tasks
	- tasks with similar rankings are considered as similar
- result on task
	- group 1: tasks rely on feature information, ie. node/graph classification -> input graphs have high node feature dim
	- group 2: tasks rely on structural information, ie. predict cluster coefficient -> few node feature dim
	- similar tasks can have similar GNN design space
- evaluating a design dimension (ie. is BN good for GNN) by controlled random search
	- sample frandom model-task configurations, perturb BN = [True, False]
	- rank BN = [True, False] by their performance
- result on design dimension
	- preprocess, postprocess how # layers, # GNN layers, # batch size, # lr depend on the task (hard to decide)
	- BN is better for GNN
	- no dropout is better for GNN
	- PreLU > ReLU > Swish for GNN
	- sum > mean > max aggregation for GNN
	- skipcat > skipsum > stack for GNN
	- Adam > SGD for GNN
	- more training epochs is better for GNN
- conclusion
	- transfer best GNN designs across similar tasks is useful
	- still, some designs may be great for some specific tasks (ie. max aggregation best for dataset BZR)

### Geometric Deep Learning (DeepMind)

