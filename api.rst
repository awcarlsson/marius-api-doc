.. _api

*************
Model, Training, Evaluation API
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

===================  ===========
   Return Type       Description
-------------------  -----------
void
===================  ===========

::

    virtual void evaluate(Batch *batch)

Runs evaluation process on specified batch.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
batch                Batch     Batch of embeddings to evaluate
===================  ========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
void
===================  ===========

::

    void save()

Save model to experiment directory specified in configuration file.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
===================  ========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
void
===================  ===========

::

    void load()

Load model from experiment directory specified in configuration file.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
===================  ========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
void
===================  ===========

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

===================  ===========
   Return Type       Description
-------------------  -----------
void                 
===================  ===========

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

===================  =
   Return Type
-------------------  -
void
===================  =

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

===================  =
   Return Type
-------------------  -
void
===================  =
