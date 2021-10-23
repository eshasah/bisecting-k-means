# Bisecting k-means algorithm implementation (text clustering)

Implement the bisecting k-Means clustering algorithm for clustering text data.

Input data (provided as training data) consists of 8580 text records in sparse format. No labels are provided. Each line in input data represents a document. Each pair of values within a line represent the term id and its count in that document.

Identified the optimal number of cluster using "Elbow graph" using Sum of Squared Error (SSE) vs No.of Clusters (k).
![kmeans graph](https://user-images.githubusercontent.com/65586427/138545479-aaeecf7c-f0d1-48a8-9867-53bc96d3e083.JPG)


Assign each of the instances in the input data to K clusters identified from 1 to K.
