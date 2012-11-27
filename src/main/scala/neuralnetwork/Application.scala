package neuralnetwork
import neuralnetwork.NeuralNetwork._
import scala.math._

object Application extends App {
    val andProgowaWeights = new FileParserNeuralNetwork("and-progowa.txt").weightsFromFile
    val andSigmoidaWeights = new FileParserNeuralNetwork("and-sigmoida.txt").weightsFromFile
    val xorSigmoidaWeights = new FileParserNeuralNetwork("xor-sigmoida.txt").weightsFromFile
    val randomWeigts = new FileParserNeuralNetwork("weights.txt").weightsFromFile

    val inputs = new InputParser("inputs.txt").inputsFromFile

    val andProgowaNN = new NeuralNetwork(andProgowaWeights)
    val andSigmoidNN = new NeuralNetwork(andSigmoidaWeights)
    val xorSigmoidNN = new NeuralNetwork(xorSigmoidaWeights)
    val randomNN = new NeuralNetwork(randomWeigts)
    
    def printList(list: List[Double]) =
        (for (elem <- list) yield elem.toString + ", ").mkString

    def printResults(nn: NeuralNetwork, inputs: List[List[Double]], title: String): Unit = {
        println(title)
        println("="*20)
        for (i <- inputs) yield {
            println("input: " + printList(i))
            println("calculated result: " + printList(nn.calculate(i)))
            println("random result: " + printList(randomNN.calculate(i)))
            println
        }
    }

    printResults(andProgowaNN, inputs, "Funkcja progowa and")
    printResults(andSigmoidNN, inputs, "Sigmoida and")
    printResults(xorSigmoidNN, inputs, "Sigmoida xor")

}
