package neuralnetwork
import neuralnetwork.NeuralNetwork._
import scala.math._

object Application extends App {
    println("Test you nerual network")

    val threshold = (x: Double) => if (x > 0) 1.0 else 0.0
    val sigmoid = (x: Double) => 1.0 / (1.0 + exp(-x))

    val andProgowaWeights = new FileParserNeuralNetwork("and-progowa.txt").weightsFromFile
    val andSigmoidaWeights = new FileParserNeuralNetwork("and-sigmoida.txt").weightsFromFile
    val xorSigmoidaWeights = new FileParserNeuralNetwork("xor-sigmoida.txt").weightsFromFile

    val inputs = new InputParser("inputs.txt").inputsFromFile

    println(xorSigmoidaWeights)
    println(inputs)

    val andProgowaNN = new NeuralNetwork(andProgowaWeights, threshold)
    val andSigmoidNN = new NeuralNetwork(andSigmoidaWeights, sigmoid)
    val xorSigmoidNN = new NeuralNetwork(xorSigmoidaWeights, sigmoid)

    println("Funkcja progowa and")
    println("="*20)
    for (i <- inputs) yield {
        println(i)
        println(andProgowaNN.calculate(i))
        println
    }

    println("Sigmoida and")
    println("="*20)
    for (i <- inputs) yield {
        println(i)
        println(andSigmoidNN.calculate(i))
        println
    }

    println("Sigmoida xor")
    println("="*20)
    for (i <- inputs) yield {
        println(i)
        println(xorSigmoidNN.calculate(i))
        println
    }
}