package thunder.streaming

import org.apache.spark.mllib.clustering.KMeansModel

/**
 * Extends clustering model for K-means with the current counts of each cluster (for streaming algorithms)
 */
class StreamingKMeansModel(override val clusterCenters: Array[Array[Double]],
                           val clusterCounts: Array[Int] = Array(1)) extends KMeansModel(clusterCenters)
