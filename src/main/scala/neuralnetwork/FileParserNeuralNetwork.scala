package neuralnetwork

import neuralnetwork.NeuralNetwork._
import io.Source.fromFile

trait StringParserNeuralNetwork {
    def weights(string: String): Weights =
        (for (line <- string.split('\n') if line != "") yield
            (for (neuronWeights <- line.split('|')) yield
                (for (weight <- neuronWeights.split(' ') if weight != "") yield
                    weight.toDouble).toList).toList).toList
}

class FileParserNeuralNetwork(val file: String) extends StringParserNeuralNetwork {
    val source = fromFile(file)
    val string = source.mkString
    source.close()
    def weightsFromFile(): Weights = {
        weights(string)
    }
}