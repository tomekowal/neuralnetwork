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

    /** epochs, LEARN_RATE, CONSCIENCE, NEIGHBOURHOOD */
    def learn(randoms : (Double, Double), parameters : List[(Int,Double,Double,Int)]) = {
        nn.randomize(randoms._1, randoms._2)
        for { (epochs, learn_rate, conscience, neigh) <- parameters } yield {
            kohonenLay.LEARN_RATE = learn_rate
            kohonenLay.CONSCIENCE = conscience
            kohonenLay.neighbourhood_dist = neigh
            nn.learn(inputs, epochs)
        }
    }

    learn((0.0, 1.0), List((8000, 0.3, 0.5, 0), (32000, 0.3, 0.3, 1)))
    printResults(nn, inputs, "Obrazki")

}
