# Text_clustering with Self-Organizing Map (SOM)
A self-organizing map (SOM) or self-organizing feature map (SOFM) is a type of artificial neural network (ANN) that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional), discretized representation of the input space of the training samples, called a map, and is therefore a method to do dimensionality reduction. Self-organizing maps differ from other artificial neural networks as they apply competitive learning as opposed to error-correction learning (such as backpropagation with gradient descent), and in the sense that they use a neighborhood function to preserve the topological properties of the input space.

# Phase 1: Document Preprocessing
1. Remove all non-letter characters from the documents.
2. Extract all words of the document and remove the short words (length ≤ 2).
3. Remove all stop words (e.g., ‘a’, ‘and’, ‘what’, ...), given in file ‘stopwords.txt’.
4. Compute the feature vector for each document, using TF-IDF weighting scheme.

# Phase 2: SOM Clustering
