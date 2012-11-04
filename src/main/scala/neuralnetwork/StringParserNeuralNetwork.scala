package neuralnetwork

import neuralnetwork.NeuralNetwork._

trait StringParserNeuralNetwork {
    def weights(string: String): Weights =
        (for (line <- string.split('\n')) yield
            (for (neuronWeights <- line.split('|')) yield
                (for (weight <- neuronWeights.split(' ')) yield
                    weight.toDouble).toList).toList).toList
}