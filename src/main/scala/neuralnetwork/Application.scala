package neuralnetwork
import neuralnetwork.NeuralNetwork._
import scala.math._

object Application extends App {
    val threshold = (x: Double) => if (x > 0) 1.0 else 0.0
    val sigmoid = (x: Double) => 1.0 / (1.0 + exp(-x))
    val line = (x: Double) => x

    val andProgowaWeights = new FileParserNeuralNetwork("and-progowa.txt").weightsFromFile
    val andSigmoidaWeights = new FileParserNeuralNetwork("and-sigmoida.txt").weightsFromFile
    val xorSigmoidaWeights = new FileParserNeuralNetwork("xor-sigmoida.txt").weightsFromFile
    val randomWeigts = new FileParserNeuralNetwork("weights.txt").weightsFromFile

    val inputs = new InputParser("inputs.txt").inputsFromFile

    /* val andProgowaNN = new NeuralNetwork(andProgowaWeights, threshold)
    val andSigmoidNN = new NeuralNetwork(andSigmoidaWeights, sigmoid)
    val xorSigmoidNN = new NeuralNetwork(xorSigmoidaWeights, sigmoid)
    val randomNN = new NeuralNetwork(randomWeigts, threshold)
    */
    def printList(list: List[Double]) =
        (for (elem <- list) yield elem.toString + ", ").mkString

    def printResults(nn: NeuralNetwork, inputs: List[List[Double]], title: String): Unit = {
        println(title)
        println("="*20)
        for (i <- inputs) yield {
            /*println("input: " + printList(i))
            println("calculated result: " + printList(nn.calculate(i)))
            println("random result: " + printList(randomNN.calculate(i)))*/
            println
        }
    }

/*    printResults(andProgowaNN, inputs, "Funkcja progowa and")
    printResults(andSigmoidNN, inputs, "Sigmoida and")
    printResults(xorSigmoidNN, inputs, "Sigmoida xor")
*/
}
