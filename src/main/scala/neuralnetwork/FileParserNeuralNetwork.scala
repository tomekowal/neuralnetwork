package neuralnetwork

import neuralnetwork.NeuralNetwork._
import io.Source.fromFile

trait StringParserNeuralNetwork {
    val reg = """^(nobias){0,1}(.*)""".r

    def weights(string: String): Weights =
        (for (line <- string.split('\n');
              val reg(nobias, weights) = line
              if line != "") yield {                      
                      val nW = (for (neuronWeights <- weights.split('|')) yield
                                         (for (weight <- neuronWeights.split(' ') if weight != "") yield
                                             weight.toDouble).toList).toList
                      nobias match {
                          case null => new BiasLayer(nW)
                          case _ => new NoBiasLayer(nW)
                     }
                  }).toList
}

class FileParserNeuralNetwork(val file: String) extends StringParserNeuralNetwork {
    val source = fromFile(file)
    val string = source.mkString
    source.close()
    def weightsFromFile(): Weights = {
        weights(string)
    }
}

