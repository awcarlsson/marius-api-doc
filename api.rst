.. _api

*************
Marius API
*************

This document describs the Marius programatic API. The API can be used in Python or C++, where there is a one-to-one mapping between the two. Implementation is done in C++, and python bindings are generated over the C++ API.

This document covers:

1. An overview of the main API objects.
2. How Marius uses the API internally to perform training and evaluation.
3. How users can define custom training routines and models.
4. Detailed API object and call information.

*************
Main API Objects
*************


Trainer/Evaluator
---------

Defines the high-level training/evaluation loop for a given input model and dataset. 


Model : torch.nn.Module
---------

The model defines all initialization of global parameters, a single iteration of training, a single iteration of evaluation, the featurizer, the GNN encoder, and the decoder. Where the featurizer, encoder, and decoder are interchangable pytorch modules. The model extends the torch.nn.module interface and supports distributed data parallel training for built-in models.

*Featurizer*: Generates new embeddings for nodes by combining node features and their respective learnable embeddings in order to emphasize individual node properties. 

*Encoder*: Generates new embeddings for nodes by aggregating node embedding with information about k-hop neighboring nodes.

*Decoder*: Takes input embeddings and produces an output score, only used for link prediction. Consists of a relation operator and a comparator.

- Relation operator: Applies edge-type embeddings to node embeddigns to produce a new node embedding with edge-type information included.
- Comparator: Compares the embeddings of a given set of source, destination, and negative embeddings to produce output scores for the postive and negative edges.

GraphBatcher
---------

The GraphBatcher exposes an iterator of batches, which defines the order in which the input data is processed. 

Batch
---------

The Batch class contains all edges, neighbors, features, and embeddings for a given batch of training or evaluation data. It is a stateful object where class members are initalized, updated, deleted at various points in the training/evaluation pipeline. 

Samplers
----------

There are four types of samplers: edge, node, negative, and neighborhood samplers. Each sampler defines how samples are produced from the graph for the given sampling item of interest. 

GraphModelStorage
----------

This class contains an interface to access the graph, node features and embeddings, and embedding optimizer state, regardless of the underlying storage backends for each type of data. 

MariusGraph 
----------

This class contains an arbitrary in-memory graph/sub-graph in CSR representation and supports fast, vectorized CPU and GPU neighbor sampling.

This class is subclassed by GNNGraph, which orders the CSR representation of the graph for fast GNN encoding.

Minor classes
---------

- Loss: Loss function to use for the model.
- Regularizer: Regularization to perform over the embeddings
- Reporter: Class used to report accuracy metrics and training/evaluation progress.
- MariusOptions: A struct shared globally by the program which contains all system configuration parameters and settings.


*************
API Usage within Marius
*************

Main Entrypoint
---------

Upon a call to `marius_train` for a given configuration file the following steps will occur:

1. Parse the input configuration file which initializes Marius with desired settings.

2. Program initialization. 

- Initialize the model used to train and evaluate the graph embeddings: Model

- Initialize the the underlying storage of the graph and embedding table: GraphModelStorage

- Define the training/evaluation sampler procedures: Samplers

- Define the ordering in which data is processed: GraphBatcher

- Define the training/evaluation epoch: Trainer/Evaluator

3. Train/evaluate for specified number of epochs.


Below shows a simplified version of the main entrypoint to Marius. 

::

	void marius(int argc, char *argv[]) {

	    marius_options = parseConfig(argc, argv); // marius_options is a global MariusOptions struct containing all program options
	    
	    Model *model = initializeModel();

	    GraphModelStorage *graph_model_storage = initializeStorage();
	    
	    EdgeSampler *edge_sampler = new RandomEdgeSampler(graph_model_storage);
	    NegativeSampler *negative_sampler = new RandomNegativeSampler(graph_model_storage);
	    NeighborSampler *neighbor_sampler = new kHopNeighborSampler(graph_model_storage, marius_options.model.num_layers, marius_options.training_sampling.neighbor_sampling_strategy, marius_options.training_sampling.max_neighbors_size);

	    GraphBatcher *graph_batcher = new GraphBatcher(graph_model_storage, edge_sampler, negative_sampler, neighbor_sampler);

	    Trainer *trainer = new SynchronousTrainer(graph_batcher, model);
	    Evaluator *evaluator = new SynchronousEvaluator(graph_batcher, model);
	    
            trainer->train(num_epochs);
	    evaluator->evaluate();
	    
	    model->save();
	    
	    // garbage collect
	}

The next sections cover the guts of the training and evaluation process: `trainer->train()` and `evaluator->evaluate()`.

Training Loop
---------

In the training loop, the Trainer will iteratively transfer batches to the GPU to calculate gradients. This process runs for the specified number of epochs. Below shows a simplified version of the synchronous training process in Marius (with timing/reporting removed).

::

	void SynchronousTrainer::train(int num_epochs) {
	
	    // set dataset to training and load edge/parameters
	    graph_batcher_->setTrainSet();
	    graph_batcher_->loadStorage();

	    for (int epoch = 0; epoch < num_epochs; epoch++) {
	    
		while (graph_batcher_->hasNextBatch()) {

		    // gets data and parameters for the next batch
		    Batch *batch = graph_batcher_->getBatch();

		    // transfers batch to the GPU
		    batch->embeddingsToDevice();

		    // compute forward and backward pass of the model
		    model_->train(batch);

		    // accumulate node embedding gradients
		    batch->accumulateGradients();

		    // transfer gradient back to host machine
		    batch->embeddingsToHost();

		    // update node embedding table and optimizer state
		    graph_batcher_->updateEmbeddingsForBatch(batch);
		}

		// notify that the epoch has been completed
		graph_batcher_->nextEpoch();
	    }
	}
	
The next two sections look closer into the model_->train(batch) function for link prediction and node classification.
	
model_->train() (Link Prediction)
---------

model_->train() (Node Classification)
---------
	
Evaluation Loop
---------

The Evaluator evaluates the generated embeddings on the validation or test set. Below is example code showing the evaluate function:

::

	void SynchronousEvaluator::evaluate(bool validation) {

	    // Set proper evaluation set
	    if (validation) {
		graph_batcher_->setValidationSet();
	    } else {
		graph_batcher_->setTestSet();
	    }
	    graph_batcher_->loadStorage();


	    // evaluation loop
	    while (graph_batcher_->hasNextBatch()) {
		Batch *batch = graph_batcher_->getBatch();
		batch->embeddingsToDevice();
		model_->evaluate(batch);
	    }
	}
	
The next two sections look closer into the model_->evaluate(batch) function for link prediction and node classification.

model_->evaluate (Link Prediction)
---------

model_->evaluate (Node Classification)
---------

*************
Classes/Functions
*************
*************
Class: Model
*************

The model is used to train and evaluate the graph embeddings. A model consists of:

1. Featurizer : Generates new embeddings for nodes by combining node features and their respective embeddings in order to emphasize individual node properties

2. Encoder : Generates new embeddings for nodes by combining node embedding with information about neighboring nodes
3. Decoder : Consists of relation operator and comparator

    - Relation operator : Encodes information about node relations/edges into embeddings
    - Comparator : Compares embeddings to generate positive and negative scores to use as input for loss function
4. Loss Function : Calculates loss for generated embeddings

5. Regularizer : Regularizes embeddings

6. Reporter : Reports on training and evaluation progress

Class Members
--------------------------
==================  ======
   Name             Type
------------------  ------
featurizer_         Featurizer
encoder_            Encoder
encoder_optimizer_  torch::optim::Optimizer
decoder_            Decoder
loss_function_      LossFunction
regularizer_        Regularizer
reporter_           Reporter
==================  ======

Functions
--------------------------
::

    virtual void train(Batch *batch)

Runs training process on specified batch.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
batch                Batch     Batch of embeddings to train on
===================  ========  ===========

::

    virtual void evaluate(Batch *batch)

Runs evaluation process on specified batch.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
batch                Batch     Batch of embeddings to evaluate
===================  ========  ===========

::

    void save()

Save model to experiment directory specified in configuration file.

::

    void load()

Load model from experiment directory specified in configuration file.

*************
Subclass: NodeClassificationModel (Model)
*************

A model designed for node classification tasks, i.e. assigning labels.

Constructor
--------------------------
::

    NodeClassificationModel(Encoder *encoder, LossFunction *loss, Regularizer *regularizer, Featurizer *featurizer, Reporter *reporter = nullptr)

Functions
--------------------------
::

    Labels forward(Batch *batch, bool train)

Forward specified batch through model to learn labels

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
batch                Batch*      The input embedding batch
train                bool        Set to true for train, false for evaluation
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
Labels               The new node labels
===================  ===========

::

    void train(Batch *batch)

Runs training process on specified batch.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
batch                Batch     Batch of embeddings to train on
===================  ========  ===========

::

    void evaluate(Batch *batch)

Runs evaluation process on specified batch.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
batch                Batch     Batch of embeddings to evaluate
===================  ========  ===========

*************
Subclass: LinkPredictionModel (Model)
*************

A model designed for link prediction tasks.

Constructor
--------------------------
::

    LinkPredictionModel(Encoder *encoder, Decoder *decoder, LossFunction *loss, Regularizer *regularizer, Featurizer *featurizer, Reporter *reporter = nullptr)

Functions
--------------------------
::

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(Batch *batch, bool train)

Forward specified batch through model to learn edges

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
batch                Batch*      The input embedding batch
train                bool        Set to true for train, false for evaluation
===================  ==========  ===========

======================================================================  ===========
   Return Type                                                          Description
----------------------------------------------------------------------  -----------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>  Updated edge embeddings
======================================================================  ===========

::

    void train(Batch *batch)

Runs training process on specified batch.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
batch                Batch     Batch of embeddings to train on
===================  ========  ===========

::

    void evaluate(Batch *batch)

Runs evaluation process on specified batch.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
batch                Batch     Batch of embeddings to evaluate
===================  ========  ===========

*************
Class: Featurizer
*************

Generates new embeddings for nodes by combining node features and their respective embeddings 
in order to emphasize individual node properties.

Functions
--------------------------
::

    virtual Embeddings operator()(Features node_features, Embeddings node_embeddings)

Combines node features with their node embeddings to generate new embeddings.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
node_features        Features    The node features
node_embeddings      Embeddings  The node embeddings
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
Embeddings           The new embeddings generated from combining input node features and node embeddings
===================  ===========

*************
Class: Encoder
*************

Generates new embeddings for nodes by combining node embedding with information about neighboring nodes.

Functions
--------------------------
::

    virtual Embeddings forward(Embeddings inputs, GNNGraph gnn_graph, bool train)

Runs encoder by passing embedding inputs through GNN.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
inputs               Embeddings  The input embeddings
gnn_graph            GNNGraph    The GNN
train                bool        Set to true for train, false for evaluation
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
Embeddings           The new embeddings updated after GNN pass-through
===================  ===========

::

    void encodeFullGraph(NeighborSampler *neighbor_sampler, GraphModelStorage *graph_storage)

Encodes graph with GNN.

===================  =================  ===========
   Parameter         Type               Description
-------------------  -----------------  -----------
neighbor_sampler     NeighborSampler    The neighborhood sampling strategy
graph_storage        GraphModelStorage  Graph model storage object
===================  =================  ===========

*************
Class: Decoder
*************

Reconstructs embedding representation of graph.

Functions
--------------------------
::

    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(Batch *, bool train)

Forwards embedding batch through Relation Operator and Comparator.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
batch                Batch       The input embedding batch
train                bool        Set to true for train, false for evaluation
===================  ==========  ===========

======================================================================  ===========
   Return Type                                                          Description
----------------------------------------------------------------------  -----------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>  The updated embeddings
======================================================================  ===========

*************
Class: RelationOperator
*************

Encodes information about node relations/edges into embeddings.

Functions
--------------------------
::

    virtual Embeddings operator()(const Embeddings &embs, const Relations &rels)

Encodes node embeddings with information about node relations.

===================  ==================  ===========
   Parameter         Type                Description
-------------------  ------------------  -----------
embs                 const Embeddings&   The input embeddings
rels                 const Relations&    The input relations
===================  ==================  ===========

===============  ===========
   Return Type   Description
---------------  -----------
Embeddings       The updated embeddings
===============  ===========

*************
Class: Comparator
*************

Compares embeddings to generate positive and negative scores to use as input for loss function.

Functions
--------------------------
::

    virtual tuple<torch::Tensor, torch::Tensor> operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs)

Takes two [n, d] tensors of embeddings as input and outputs a score/distance metric for each element-wise pair.

===================  ==================  ===========
   Parameter         Type                Description
-------------------  ------------------  -----------
src                  const Embeddings&   Source embeddings
dst                  const Embeddings&   Destination embeddings
negs                 const Embeddings&   Negative samples
===================  ==================  ===========

===================================  ===========
   Return Type                       Description
-----------------------------------  -----------
tuple<torch::Tensor, torch::Tensor>  Positive and negative scores
===================================  ===========

*************
Class: LossFunction
*************

Calculates loss for generated embeddings.

Functions
--------------------------
::

    virtual torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores)

Takes positive and negative scores and calculates loss.

===================  ==================  ===========
   Parameter         Type                Description
-------------------  ------------------  -----------
pos_scores           torch::Tensor       Positive scores
neg_scores           torch::Tensor       Negative scores
===================  ==================  ===========

================  ===========
   Return Type    Description
----------------  -----------
torch::Tensor     Loss
================  ===========

*************
Class: Regularizer
*************

*************
Class: Reporter
*************

*************
Class: Trainer
*************

The trainer runs the training process using the given model for the specified number of epochs.

Class Members
--------------------------
==================  ======
   Name             Type
------------------  ------
graph_batcher_      GraphBatcher
progress_reporter_  ProgressReporter
model_              Model
==================  ======

Functions
--------------------------
::

    virtual void train(int num_epochs = 1)

Runs training process for embeddings for specified number of epochs.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
num_epochs           int       The number of epochs to train for
===================  ========  ===========

*************
Class: Evaluator
*************

The evaluator runs the evaluation process using the given model.

Class Members
--------------------------
==================  ======
   Name             Type
------------------  ------
graph_batcher_      GraphBatcher
model_              Model
==================  ======

Functions
--------------------------
::

    virtual void evaluate(bool validation)

Runs evaluation process.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
validation           bool      If true, evaluate on validation set. Otherwise evaluate on test set
===================  ========  ===========

*************
Class: GraphBatcher
*************
Represents a training or evaluation set for graph embedding. Iterates over batches and updates model parameters during training.

Class Members
--------------------------
==================  ======
   Name             Type
------------------  ------
graph_storage_      GraphModelStorage
neighbor_sampler_   NeighborSampler
==================  ======

Constructor
--------------------------
::

    GraphBatcher(GraphModelStorage *graph_storage, EdgeSampler *edge_sampler, NegativeSampler *negative_sampler, NeighborSampler *training_neighbor_sampler, NeighborSampler *evaluation_neighbor_sampler = nullptr)


Functions
--------------------------
::

    void setTrainSet()

Sets graph storage, negative sampler, and neighbor sampler to training set.

::

    void setValidationSet()

Sets graph storage, negative sampler, and neighbor sampler to validation set.

::

    void loadStorage()

Load graph from storage.

::

    void unloadStorage(bool write = false)

Unload graph from storage.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
write                bool        Set to true to write graph state to disc
===================  ==========  ===========

::

    int64_t getEpochsProcessed()

Get the number of epochs processed.

===================  ===========
   Return Type       Description
-------------------  -----------
int64_t              Number of epochs processed                
===================  ===========

::

    bool hasNextBatch()

Check to see whether another batch exists.

===================  ===========
   Return Type       Description
-------------------  -----------
bool                 True if batch exists, false if not             
===================  ===========

::

    Batch *getBatch()

Gets the next batch to be processed by the pipeline. Loads edges from storage, constructs negative edges, and loads CPU embedding parameters.

===================  ===========
   Return Type       Description
-------------------  -----------
Batch*               The next batch          
===================  ===========

::

    void loadGPUParameters(Batch *batch, bool encoded=false)

Loads GPU parameters into batch.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
batch                Batch*      Batch object to load parameters into
encoded              bool        True for encoded, false if not
===================  ==========  ===========

::

    void updateEmbeddingsForBatch(Batch *batch, bool gpu)

Applies gradient updates to underlying storage.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
batch                Batch*      Batch object to apply updates from
gpu                  bool        If true, only the gpu parameters will be updated
===================  ==========  ===========

::

    void finishedBatch()

Notify that the batch has been completed.

::

    void nextEpoch()

Notify that the epoch has been completed. Prepares dataset for a new epoch.

::

    int64_t getNumEdges()

Gets the number of edges from the graph storage.

===================  ===========
   Return Type       Description
-------------------  -----------
int64_t              Number of edges in the graph                           
===================  ===========

*************
Class: Batch
*************
Contains metadata, edges and embeddings for a single batch.


Constructor
--------------------------
::

    Batch(bool train)


Functions
--------------------------
::

    void localSample()

Construct additional negative samples and neighborhood information from the batch.

::

    void embeddingsToDevice(int device_id)

Transfers embeddings, optimizer state, and indices to specified device.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
device_id            int         Device id to transfer to
===================  ==========  ===========

::

    void prepareBatch()

Populates the src_pos_embeddings, dst_post_embeddings, relation_embeddings, src_neg_embeddings, and dst_neg_embeddings tensors for model computation.

::

    void accumulateGradients()

Accumulates gradients into the unique_node_gradients and unique_relation_gradients tensors, and applies optimizer update rule to create the unique_node_gradients2 and unique_relation_gradients2 tensors.

::

    void embeddingsToHost()

Transfers gradients and embedding updates to host.

::

    void clear()

Clears all tensor data in the batch.

*************
Class: EdgeSampler
*************

Samples the edges from a given batch.

Class Members
--------------------------
==================  ======
   Name             Type
------------------  ------
graph_storage_      GraphModelStorage*
==================  ======

Functions
--------------------------
::

    virtual EdgeList getEdges(Batch *batch)

Get edges from the given batch.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
batch                Batch*      Batch to sample from
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
EdgeList             The edges from the batch
===================  ===========

*************
Class: NegativeSampler
*************

Samples the negative edges from a given batch.

Class Members
--------------------------
==================  ======
   Name             Type
------------------  ------
graph_storage_      GraphModelStorage*
sampler_lock_       std::mutex
==================  ======

Functions
--------------------------
::

    virtual torch::Tensor getNegatives(Batch *batch, bool src)

Get negative edges from the given batch. Returns tensor of node IDs of shape [num_negs] or a [num_negs, 3] shaped tensor of negative edges.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
batch                Batch*      Batch to sample from
src                  bool        Source
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
EdgeList             The negative edges from the batch
===================  ===========

::

    void lock()

Lock the sampler_lock_.

::

    void unlock()

Unlock the sampler_lock_.

*************
Class: NeighborSampler
*************

Samples the neighbors from a given batch given a neighbor sampling strategy.

Class Members
--------------------------
===========================  ======
   Name                      Type
---------------------------  ------
graph_storage_               GraphModelStorage*
sampler_lock_                std::mutex
neighbor_sampling_strategy_  NeighborSamplingStrategy
max_neighbors_size_          int
===========================  ======

Functions
--------------------------
::

    virtual GNNGraph getNeighbors(torch::Tensor node_ids, bool incoming, bool outgoing)

Get neighbors of provided nodes using given neighborhood sampling strategy.

===================  =============  ===========
   Parameter         Type           Description
-------------------  -------------  -----------
node_ids             torch::Tensor  Nodes to get neighbors from
incoming             bool           True if including incoming neighbors
outgoing             bool           True if including outgoing neighbors
===================  =============  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
GNNGraph             The neighbors sampled using strategy
===================  ===========

::

    void lock()

Lock the sampler_lock_.

::

    void unlock()

Unlock the sampler_lock_.

*************
Class: MariusGraph
*************

Object to handle arbitrary in-memory graph/sub-graph.

Class Members
--------------------------
==========================  ======
   Name                     Type
--------------------------  ------
src_sorted_edges_           EdgeList
dst_sorted_edges_           EdgeList
active_in_memory_subgraph_  EdgeList
node_ids_                   Indices
out_sorted_uniques_         Indices
out_offsets_                Indices
out_num_neighbors_          torch::Tensor
in_sorted_uniques_          Indices
in_offsets_                 Indices
in_num_neighbors_           torch::Tensor
==========================  ======

Constructor
--------------------------
::

    MariusGraph()
    MariusGraph(EdgeList edges)

Functions
--------------------------
::

    Indices getNodeIDs()

Get the node IDs from the graph.

===================  ===========
   Return Type       Description
-------------------  -----------
Indices              Tensor of node IDs
===================  ===========

::

    Indices getEdges(bool incoming = true)

Get the edges from the graph.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
incoming             bool        Get incoming edges if true, outgoing edges if false
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
Indices              Tensor of edge IDs
===================  ===========

::

    Indices getRelationIDs(bool incoming = true)

Get the relation IDs from the graph.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
incoming             bool        Get incoming relation IDs if true, outgoing relation IDs if false
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
Indices              Tensor of relation IDs
===================  ===========

::

    Indices getNeighborOffsets(bool incoming = true)

Get the neighbor offsets from the graph.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
incoming             bool        Get incoming neighbor offsets if true, outgoing neighbor offsets if false
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
Indices              Tensor of neighbor offsets
===================  ===========

::

    torch::Tensor getNumNeighbors(bool incoming = true)

Get the number of neighbors in a graph.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
incoming             bool        Get number of incoming neighbor if true, number of outgoing neighbors if false
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
torch::Tensor        Number of neighbors
===================  ===========

::

    std::tuple<torch::Tensor, torch::Tensor> getNeighborsForNodeIds(torch::Tensor node_ids, bool incoming, NeighborSamplingStrategy neighbor_sampling_strategy, int max_neighbors_size)

Get the neighbors for the specified node IDs.

==========================  ========================  ===========
   Parameter                Type                      Description
--------------------------  ------------------------  -----------
node_ids                    torch::Tensor             The node IDs to get neighbors from
incoming                    bool                      Get incoming neighbors if true, outgoing if false
neighbor_sampling_strategy  NeighborSamplingStrategy  The neighbor sampling strategy to use
max_neighbors_size          int                       The maximum number of neighbors to sample
==========================  ========================  ===========

========================================  ===========
   Return Type                            Description
----------------------------------------  -----------
std::tuple<torch::Tensor, torch::Tensor>  Neighbors of specified nodes.
========================================  ===========

::

    void clear()

Clear the graph.

*************
Subclass: GNNGraph (MariusGraph)
*************

MariusGraph sublass, orders the CSR representation of the graph for fast GNN encoding.

Class Members
--------------------------
==========================  ======
   Name                     Type
--------------------------  ------
hop_offsets_                Indices
in_neighbors_mapping_       Indices
out_neighbors_mapping_      Indices
in_neighbors_vec_           std::vector<torch::Tensor>
out_neighbors_vec_          std::vector<torch::Tensor>
num_nodes_in_memory_        int
==========================  ======

Constructor
--------------------------
::

    GNNGraph()
    GNNGraph(Indices hop_offsets, Indices node_ids, Indices in_offsets, std::vector<torch::Tensor> in_neighbors_vec, Indices in_neighbors_mapping, Indices out_offsets, std::vector<torch::Tensor> out_neighbors_vec, Indices out_neighbors_mapping, int num_nodes_in_memory)

Functions
--------------------------
::

    void prepareForNextLayer()

Prepares GNN graph for next layer.

::

    Indices getNeighborIDs(bool incoming = true, bool global = false)

Get the neighbor IDs.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
incoming             bool        Get incoming edges if true, outgoing edges if false
global               bool        If false, return node IDs local to the batch. If true, return any global node IDs
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
Indices              Tensor of edge IDs
===================  ===========

::

    int64_t getLayerOffset()

Gets the layer offset.

===================  ===========
   Return Type       Description
-------------------  -----------
int64_t              Layer offset
===================  ===========

::

    void performMap()

Maps local IDs to batch.
