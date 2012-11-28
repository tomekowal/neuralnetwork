package neuralnetwork
import neuralnetwork.NeuralNetwork._
import scala.math._

object AppKohonen extends App {
    val inputs = new InputParser("koh.txt").inputsFromFile

    val kohonenLay = new KohonenLayer(9, 4)
    val nn = new NeuralNetwork(List(kohonenLay))
    
    def printList(list: List[Double]) =
        (for (elem <- list) yield elem.toString + ", ").mkString

    def printResults(nn: NeuralNetwork, inputs: List[List[Double]], title: String): Unit = {
        println(title)
        println("="*20)
        for (i <- inputs) yield {
            println("input: " + printList(i))
            println("calculated result: " + printList(nn.calculate(i)))

            println
        }
    }

    def learn() = {
        nn.randomize(0.0, 0.1)
        nn.learn(inputs, 32000)       
    }

    learn
    printResults(nn, inputs, "Obrazki")

}
