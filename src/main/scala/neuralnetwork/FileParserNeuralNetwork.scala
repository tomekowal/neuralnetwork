package neuralnetwork

import neuralnetwork.NeuralNetwork._
import io.Source.fromFile

trait StringParserNeuralNetwork {
    val reg = """^(nobias){0,1}\s*(linear|l|sigmoid|s|thresh|t)(.*)""".r

    def weights(string: String): Weights =
        (for (line <- string.split('\n');
              val reg(nobias, activation, weights) = line
              if line != "") yield {                      
                      val nW = (for (neuronWeights <- weights.split('|')) yield
                                         (for (weight <- neuronWeights.split(' ') if weight != "") yield
                                             weight.toDouble).toList).toList
		      val bias = if (nobias == null) true else false
                      activation match {
                          case "linear" | "l" => new LinearLayer(nW, bias)
			  case "sigmoid" | "s" => new SigmoidLayer(nW, bias)
			  case "thresh" | "t" => new ThresholdLayer(nW, bias)
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

