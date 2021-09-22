.. _api

*************
Model, Training, Evaluation API
*************

*************
General Training and Evaluation Process
*************

Marius is a system under active development for training embeddings for large-scale graphs on a single machine. The general outline for training and evaluating is as follows:

1. Parse the input configuration file which initializes Marius with desired settings

2. Initialize the model used to train and evaluate the graph embeddings

3. Initialize the graph model storage

4. Initialize the edge, negative, training neighbor, and evaluation neighbor samplers

5. Initialize the graph batcher

6. Initialize the trainer or evaluator

7. Train/evaluate for specified number of epochs

Below is example code showing a typical end-to-end training/evaluation process:

::

	void marius(int argc, char *argv[]) {

	    marius_options = parseConfig(argc, argv);

	    bool train = true;
	    string path = string(argv[0]);
	    string base_filename = path.substr(path.find_last_of("/\\") + 1);
	    if (strcmp(base_filename.c_str(), "marius_eval") == 0) {
		train = false;
	    }

	    if (!train) {
		marius_options.storage.reinitialize_edges = false;
		marius_options.storage.reinitialize_embeddings = false;
	    }

	    torch::manual_seed(marius_options.general.random_seed);

	    Timer preprocessing_timer = Timer(false);
	    preprocessing_timer.start();
	    SPDLOG_INFO("Start preprocessing");
	    
	    Model *model = initializeModel();

	    GraphModelStorage *graph_model_storage = initializeStorage(train);
	    
	    EdgeSampler *edge_sampler = new RandomEdgeSampler(graph_model_storage);
	    NegativeSampler *negative_sampler = nullptr;
	    NeighborSampler *training_neighbor_sampler = nullptr;
	    NeighborSampler *evaluation_neighbor_sampler = nullptr;

	    if (marius_options.general.learning_task == LearningTask::LinkPrediction) {
		negative_sampler = new RandomNegativeSampler(graph_model_storage);
	    }

	    if (marius_options.model.encoder_model != EncoderModelType::None) {
		training_neighbor_sampler = new kHopNeighborSampler(graph_model_storage, marius_options.model.num_layers, marius_options.training_sampling.neighbor_sampling_strategy, marius_options.training_sampling.max_neighbors_size);
		evaluation_neighbor_sampler = new kHopNeighborSampler(graph_model_storage, marius_options.evaluation.num_layers, marius_options.evaluation.neighbor_sampling_strategy, marius_options.evaluation.max_neighbors_size);
	    }

	    GraphBatcher *graph_batcher = new GraphBatcher(graph_model_storage, edge_sampler, negative_sampler, training_neighbor_sampler, evaluation_neighbor_sampler);

	    preprocessing_timer.stop();
	    int64_t preprocessing_time = preprocessing_timer.getDuration();

	    SPDLOG_INFO("Preprocessing Complete: {}s", (double) preprocessing_time / 1000);

	    Trainer *trainer;
	    Evaluator *evaluator;

	    if (train) {
		if (marius_options.training.synchronous) {
		    trainer = new SynchronousTrainer(graph_batcher, model);
		} else {
		    trainer = new PipelineTrainer(graph_batcher, model);
		}

		if (marius_options.evaluation.synchronous) {
		    evaluator = new SynchronousEvaluator(graph_batcher, model);
		} else {
		    evaluator = new PipelineEvaluator(graph_batcher, model);
		}

		for (int epoch = 0; epoch < marius_options.training.num_epochs; epoch += marius_options.evaluation.epochs_per_eval) {
		    int num_epochs = marius_options.evaluation.epochs_per_eval;
		    if (marius_options.training.num_epochs < num_epochs) {
		        num_epochs = marius_options.training.num_epochs;
		        trainer->train(num_epochs);
		    } else {
		        trainer->train(num_epochs);
		        evaluator->evaluate(true);
		    }
		}
		evaluator->evaluate(false);

		model->save();

	    } else {
		if (marius_options.evaluation.synchronous) {
		    evaluator = new SynchronousEvaluator(graph_batcher, model);
		} else {
		    evaluator = new PipelineEvaluator(graph_batcher, model);
		}
		evaluator->evaluate(false);
	    }

	    // garbage collect
	    delete graph_model_storage;
	    delete trainer;
	    delete evaluator;
	    delete graph_batcher;
	}

*************
Training Loop
*************

In the training loop, the specified Trainer will iteratively transfer batches to the GPU to calculate gradients. This process runs for the specified number of epochs. Below is example code showing a train function:

::

	void SynchronousTrainer::train(int num_epochs) {
	    graph_batcher_->setTrainSet();
	    graph_batcher_->loadStorage();
	    Timer timer = Timer(false);

	    for (int epoch = 0; epoch < num_epochs; epoch++) {
		timer.start();
		SPDLOG_INFO("################ Starting training epoch {} ################", graph_batcher_->getEpochsProcessed() + 1);
		while (graph_batcher_->hasNextBatch()) {

		    // gets data and parameters for the next batch
		    Batch *batch = graph_batcher_->getBatch();

		    // transfers batch to the GPU
		    batch->embeddingsToDevice(0);

		    // loads model parameters that reside in the GPU
		    graph_batcher_->loadGPUParameters(batch);

		    // compute forward and backward pass of the model
		    model_->train(batch);

		    // transfer gradients and update parameters
		    if (batch->unique_node_embeddings_.defined()) {
		        batch->accumulateGradients();
		        batch->embeddingsToHost();

		        graph_batcher_->updateEmbeddingsForBatch(batch, true);
		        graph_batcher_->updateEmbeddingsForBatch(batch, false);
		    }

		    // notify that the batch has been completed
		    graph_batcher_->finishedBatch();

		    // log progress
		    progress_reporter_->addResult(batch->batch_size_);
		}
		SPDLOG_INFO("################ Finished training epoch {} ################", graph_batcher_->getEpochsProcessed() + 1);

		// notify that the epoch has been completed
		graph_batcher_->nextEpoch();
		progress_reporter_->clear();
		timer.stop();

		std::string item_name;
		int64_t num_items = 0;
		if (marius_options.general.learning_task == LearningTask::LinkPrediction) {
		    item_name = "Edges";
		    num_items = graph_batcher_->getNumEdges();
		} else if (marius_options.general.learning_task == LearningTask::NodeClassification) {
		    item_name = "Nodes";
		    num_items = marius_options.general.num_train;
		}

		int64_t epoch_time = timer.getDuration();
		float items_per_second = (float) num_items / ((float) epoch_time / 1000);
		SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
		SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);

		if (marius_options.model.encoder_model != EncoderModelType::None && marius_options.general.learning_task == LearningTask::LinkPrediction) {
		    model_->encoder_->encodeFullGraph(graph_batcher_->neighbor_sampler_, graph_batcher_->graph_storage_);
		}

	    }
	    graph_batcher_->unloadStorage(true);
	}
	
*************
Evaluation
*************

The Evaluator evaluates the generated embeddings on the validation or test set. Below is example code showing an evaluate function:

::

	void SynchronousEvaluator::evaluate(bool validation) {

	    if (validation) {
		graph_batcher_->setValidationSet();
	    } else {
		graph_batcher_->setTestSet();
	    }

	    graph_batcher_->loadStorage();

	    bool encoded = false;
	    if (marius_options.model.encoder_model != EncoderModelType::None) {
		encoded = true;
	    }

	    Timer timer = Timer(false);
	    timer.start();
	    int num_batches = 0;
	    while (graph_batcher_->hasNextBatch()) {
		Batch *batch = graph_batcher_->getBatch(); // gets the node embeddings and edges for the batch
		batch->embeddingsToDevice(0); // transfers the node embeddings to the GPU
		graph_batcher_->loadGPUParameters(batch, encoded); // load the edge-type embeddings to batch
		model_->evaluate(batch);
		graph_batcher_->finishedBatch();
		num_batches++;
	    }
	    timer.stop();

	    model_->reporter_->report();

	    graph_batcher_->unloadStorage();
	}

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

===================  ===========
   Return Type       Description
-------------------  -----------
void
===================  ===========

::

    void evaluate(Batch *batch)

Runs evaluation process on specified batch.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
batch                Batch     Batch of embeddings to evaluate
===================  ========  ===========

===================  =
   Return Type
-------------------  -
void
===================  =

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

===================  ===========
   Return Type       Description
-------------------  -----------
void
===================  ===========

::

    void evaluate(Batch *batch)

Runs evaluation process on specified batch.

===================  ========  ===========
   Parameter         Type      Description
-------------------  --------  -----------
batch                Batch     Batch of embeddings to evaluate
===================  ========  ===========

===================  =
   Return Type
-------------------  -
void
===================  =

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

Constructor
--------------------------
::

    GraphBatcher(GraphModelStorage *graph_storage, EdgeSampler *edge_sampler, NegativeSampler *negative_sampler, NeighborSampler *training_neighbor_sampler, NeighborSampler *evaluation_neighbor_sampler = nullptr)


Functions
--------------------------
::

    void setTrainSet()

Sets graph storage, negative sampler, and neighbor sampler to training set.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
void                 
===================  ===========

::

    void setValidationSet()

Sets graph storage, negative sampler, and neighbor sampler to validation set.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
void                 
===================  ===========

::

    void loadStorage()

Load graph from storage.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
void                 
===================  ===========

::

    void unloadStorage(bool write = false)

Unload graph from storage.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
write                bool        Set to true to write graph state to disc
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
void                 
===================  ===========

::

    int64_t getEpochsProcessed()

Get the number of epochs processed.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
int64_t              Number of epochs processed                
===================  ===========

::

    bool hasNextBatch()

Check to see whether another batch exists.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
bool                 True if batch exists, false if not             
===================  ===========

::

    Batch *getBatch()

Gets the next batch to be processed by the pipeline. Loads edges from storage, constructs negative edges, and loads CPU embedding parameters.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
===================  ==========  ===========

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

===================  ===========
   Return Type       Description
-------------------  -----------
void                           
===================  ===========

::

    void updateEmbeddingsForBatch(Batch *batch, bool gpu)

Applies gradient updates to underlying storage.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
batch                Batch*      Batch object to apply updates from
gpu                  bool        If true, only the gpu parameters will be updated
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
void                           
===================  ===========

::

    void finishedBatch()

Notify that the batch has been completed.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
void                           
===================  ===========

::

    void nextEpoch()

Notify that the epoch has been completed. Prepares dataset for a new epoch.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
void                           
===================  ===========

::

    int64_t getNumEdges()

Gets the number of edges from the graph storage.

===================  ==========  ===========
   Parameter         Type        Description
-------------------  ----------  -----------
===================  ==========  ===========

===================  ===========
   Return Type       Description
-------------------  -----------
int64_t              Number of edges in the graph                           
===================  ===========
