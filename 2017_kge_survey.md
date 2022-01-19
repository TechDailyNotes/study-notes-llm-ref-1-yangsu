## Source

- [Knowledge Graph Embedding: A Survey of Approaches and Applications](https://persagen.com/files/misc/Wang2017Knowledge.pdf)

## KG embedding with facts alone

### steps

- represent entities and relations
- definte *scoring function*
- learn entitiy and relation representations

### translational distance models (distance based scoring functions)

- TransE and its extensions
    - TransE: euclidean distance of h and t (h + r - t)
    - [introduce relation-specific entity embeddings]
    - TransH: r norm projection hyperplane of h and t
    - TransR: r projection matrix of h and t
    - TransD: matrix of mapping vector multiplication of h,r,t
    - TranSparse: TransR with sparseness on projection matrix
    - [relaxing translational requirement h+r=t]
    - TransM: TransE with weights
    - ManifoldE: TransE with a hyper-sphere center ((h + r - t) - theta)
    - TransF: relax to direction similarity of h and t
    - TransA: TransE with *Mahalanobis distance*
- Gaussian Embeddings
    - [model h,r,t as random variables instead of deterministic points in vector spaces]
    - KG2E: Gaussian, distance measure includes *Kullback-Leibler divergence* and *probability inner product*
    - TransG: mixture Gaussians, can learn sematic components automatically by *Chinese restaurant process*
- Unstructured model (UM): TransE with r=0 -> does not distinguish different relations
- Structured embedding (SE): uses two projection matrix for h,t respectively

### semantic matching models (similarity based scoring functions)

- RESCAL and its extensions
    - RESCAL: matrix interactions of h and t (h * M_r * t)
    - TATEC: RESCAL with 2-way interactions (h * r, t * r, h * D_r * t)
    - DistMult: simplified RESCAL with diagonal M_r -> pairwise interactions of h and t along the same dimension
    - HolE (Holographic Embeddings): DistMult with *circular correlation operation* -> compositional interactions of h and t along the same dimension
    - ComplEx (Complex Embeddings): DistMult with complex values (proved to be general case of HolE)
    - ANALOGY: RESCAL with normality and commutativity restrictions (proved to be general case of DistMult, HolE, ComplEx)
- Matching with Neural Networks
    - SME (Semantic Matching Energy): weighted matrices of (h,r) and (t,r), then + or *
    - NTN (Neural Tensor Network): weighted matrices of h, t and (h,t), then +, tanh and * r -> expensive, also when (h,t) matrices and bias = 0 -> SLM (single layer model)
    - MLP (Multi-Layer Perceptron): weighted matrices of h, t and r, then +, tanh and * w
    - NAM (Naural Association Model): MLP on concatenation of h and r, then * t, also has a mmore complicated version where r gets inputted in every layer of MLP as well

### model training

- under OWA (open world assumption): KG has only true facts
    - KG -> *+ examples*
    - use local CWA (close world assumption) -> *- examples*
    - minimize logistic loss -> find transitive relations (shown to be good for similarity based models)
    - minimize pairwise ranking loss -> *+ examples* scores higher than *- examples* (shown to be good for distance based models)
    - both can use minibatch SGD to optimize
    - initialization: uniform or Gaussian random, or result from TransE, or name embedding by word pretraining
    - generate *- examples*: replace h,r,t by a random one -> may introduce false-negative examples
    - better: probabilistic replacement by number of h and t; or for each r, only use h and t that appear to be entitiy of r
- under CWA (close world assumption): all facts that are not in *+ examples* (unobserved) are false
    - minimize squared loss, or logistic loss, or absolute loss
    - the optimization of RESCAL becomes a *tensor factorization problem*, can be solved by ALS (alternating least squares), or CANDECOMP/PARAFAC decomposition, or Tucker decomposition
    - disadvantages: there may be a lot of unobserved facts, so perform worse than OWA (shown in downstream tasks); also create ctoo many *- examples*, leading to scalabiity issues

### model comparison

- models that represent h,r,t as vectors are more effecient, eg. TransE, TransH, DistMult, ComplEx
- models that represent r as matrices have higher complexity, eg. TransR, SE, RESCAL; or tensors, eg. NTN -> because they scale quadratically or cubically with the dim of embedding space
- models based on neural networks have higher complexity in time (if not space) because of matrix / tensor computations
- in link prediction tasks accuracy: (ANALOGY > HolE > ) TransE and DistMult > MLP > NTN (so simple models are better, NN overfits small datasets)

### other approaches

- for each (h,r,t), embedds (h,t) pair to vector p, then do pairwise ranking loss minimization on (p,r) -> so learn pairwise representation instead of individual entities

## incorporating additional information

### entity types

- simple: if some entities belongs to one group, denote it by (h, *IsA*, h')
- SSE (sematically smooth embedding): forces entities of the same group to be close in their embedding space -> SSE uses *Laplacian eigenmaps* and *locally linear embedding* to model the above smoothness measure
- SSE performs better than the simple one, but it limits/assumes each h belongs to exactly one group (which is not the case in real world KG)
- TKRL (type-embodied knowledge representation learning): can handle multiple group labels, given (h,r,t), it projects h and t with type-specific matrices -> (M_rh * h + r - M_rt * t), where M_rh is the weighted sum of all possible type matrices -> good performance but high time and space complexity
- or add entitiy type restrictions to *- examples* generation when training, or generate when low %

### relation paths

- (h, r1, r2, ..., rl, t), multi-hop relationships -> (h, r, t) or (h, p, t)
- people usually use composition of rs to represent the path p from h to t: sum(rs) or multi(rs) or RNN(rs)
- PTransE (path-based TransE): consider all 3 (sum, multi, RNN), the path p is compared to a direct relation r, and (p-r) should be small if (h,r,t) holds -> defines a reliability function to measure it, using a *network-based resource-allocation mechanism*, then aggregate all path and combine it with the original TransE loss
- TransE + RESCAL: (h + [r1 + r2 + ... + rl] - t) and (h * [M1 * M2 * ... * Ml] * t) -> perform well in QA on KGs
- all has huge complexity issue
- another: use dynamic programming -> path of bounded length, also entities are modeled in the path

### textual descriptions

- back in NTN textual info is used to initialize the embeddings of entities
- can use average of entities names representations (learned prior from a corpus) or description representations
- but this approach models textual info separately from KG facts, hence fails
- joint model
    - knowledge model: a variant of TransE, measure fitness to KG facts
    - text model: a variant of skip-gram, embeds words in corpus, measure fitness to co-occurring word pairs
    - alignment model: make sure knowledge model and text model lie in the same space
    - target is to minimize the loss of sum of all 3 above
    - alignment can enhance the above 2, can also enable prediction of out-of-KG entities (text in corpus but not in KG yet)
- DKRL (description-embodied knowledge representation learning): TransE + word embedding (which can be a bag-of-word or CNN encoder) on entities, perform better than TransE particularly in out-of-KG entities
- TEKE (text-enhanced KG embedding model)
    - context embedding: define textual context of h,t as the co-occurenced neighbors in the corpus and use common neighbors of h,t to define the neighbors of r
    - the new h',r',t' is then a matrix projection of its context embedding N (h' = M * N + h)
    - outperform original model (which w/o context embedding)

### logical rules

- first-order Horn clauses: l1(x, y) -> l2(x, y), if (x, l1, y) then (x, l2, y)
    - well studied in *Markov logic networks*
    - also has WARMR, ALEPH, AMIE which extract logical rules from KG automatically
- one: use rules to refine embedding models during KG completion -> linear programming problem
- KALE (key ingredient of logical rule embedding): represents facts and rules in atomic way
- one similar to KALE: represent pair-wise embeddings instead
- both drawback: atomic groud truth too expensive -> some work use regularization to limit the complexity, but only work for simple rules, not generalizable

### Other Information

- entity attributes
    - relations between entities and relations between entity and attribute should be separate
    - define a entity-attribute matrix, factorized jointly with the (h,r,t) embedding tensor
- temporal information
    - KG are time-awared, propose a time-aware embedding model
    - for two relations ri, rj with temporal transitions, there should be a matrix M captures this transition, and M * ri should be close to rj
    - another: use True/False indicator to denote when the relation is established and vanished, (h,r,t,s) where s = vector representing time stamp, see *dynamic KGs* -> perform well in dynamic domains (medical, sensor data)
    - another: uses a *temporal point process* to model occurrence of facts, learn non-linearly evolving entity representations
- graph structures
    - 3 structures: normal KG, relation path KG, edge KG; the last one is new, edge is defined by all entities linking to and from it
    - other: estimate the plausibility of a fact (h,r,t) from # of facts where h, t is head, tail, # of facts where (h,-,t) holds and the relation can be any
    - show good performance in link prediction
- evidence from other relational learning methods
    - KG (MLP) + PRA (page ranking algorithm)
    - KG (RESCAL) + PRA

# Applications in Downstream Tasks

### In-KG Applications

- link prediction (entity prediction / ranking)
    - (h,r,?) or (?,r,t), is a KG completion task
    - also relation prediction (h,?,t)
    - can be done by a score ranking of every h' where (h',r,t) or t' where (h,r,t')
    - evaluation: lower rank of correct h' is better -> metrics: mean rank, mean reciprocal rank, Hits@n, AUC-PR
- triple classification
    - is an unseen (h,r,t) true or false, is a KG completion task
    - take a threshold th_r, then predict true if the scoring function on (h,r,t) > th_r
    - evaluation metrics: micro- and macro- averaged accuracy, mean average precision
- entity classification
    - which group does h belongs to -> can be treated as link prediction (h, *isA*, ?)
    - evaluation: as above
- entity resolution
    - does h == t
    - can also be treated as triple classification (h, *is*, t) true or false, but does not always work when *is* is not there for some h,t; can also set likelihood score to the similarity of h and t, which works

### Out-of-KG Applications

- relation extraction
    - given a sentence with detected h and t, find r
    - previous approach uses only *text-based extractors*, ignoring the effect of KG
    - one: combine KG relations of h, t and text-based extractors, assign a composite score for the candidate fact (r)
    - another: set occurrence matrix of text-mentions and KG relations, task is to predict missing KG relations given text-mentions
    - others also use matrix completion techniques instead of matrix factorization
    - there are variants which encodes test and KG in 3-mode tensor and factorize using RESCAL
- question answering (over KGs)
    - given question, retrieve the correct (h,r,t) fact
    - measure the similarity of embeddings of question q and answer a ([W * v(q)] * [W * v(a)]), where v(.) is the vector representation and W is the word embedding matrix
    - can optimize by pariwise ranking optimization
    - data can be generated by *crowdsourcing* or generalizing seed patterns over KGs
    - measure score is higher if answer is more correct
- recommender systems
    - previous approach uses *collaborative filtering techniques*, ignoring the effect of KG
    - *hybrid recommender systems* combine user-item interactions and info of user and item perform better
    - one: uses (user, r, item) facts -> KG embedding; textual -> *de-noising auto-encoders*;and visual -> *convolutional auto-encoders* knowledge of item to derive semantic representation of items
    - score preference of user to item by u * (s + t + v + n), where u is user, s,t,v are structural, textual and visual representation of item, n is offset vector
    - item recommendation can be made by score ranking of items
