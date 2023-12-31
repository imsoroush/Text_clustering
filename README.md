# Text_clustering with Self-Organizing Map (SOM)
A self-organizing map (SOM) or self-organizing feature map (SOFM) is a type of artificial neural network (ANN) that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional), discretized representation of the input space of the training samples, called a map, and is therefore a method to do dimensionality reduction. Self-organizing maps differ from other artificial neural networks as they apply competitive learning as opposed to error-correction learning (such as backpropagation with gradient descent), and in the sense that they use a neighborhood function to preserve the topological properties of the input space.

# Phase 1: Document Preprocessing
1. Remove all non-letter characters from the documents.
2. Extract all words of the document and remove the short words (length ≤ 2).
3. Remove all stop words (e.g., ‘a’, ‘and’, ‘what’, ...), given in file ‘stopwords.txt’.
4. Compute the feature vector for each document, using TF-IDF weighting scheme.

# Phase 2: SOM Clustering
a) Winner-takes-all approach

for this part first randomly initialize weight vector which is size =
(num_neurons, num_data). then set the learning parameters.
in each epoch all datas are chosen to train the model. but the order of
choosing data is random. for each data we find distance between data
and output neurons. and find closest to data as k. then update winner
neuron(k) based on its distance to data. the learning iterations goes till
max_epoch reached or when largest change(max norm change in w in previous
epoch) is less than a threshold. for prediction we find distance of each
sample to each neuron and set label of closest to data.


b) On-center, off-surround approach

this part is much like the previous one, the difference is in output neurons.
for example if we have 3*3 output neurons, in every iteration furthermore
winner’s weights, also neighbour’s weights will update base on
their distance to winner.
