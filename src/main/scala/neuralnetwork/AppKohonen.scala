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

    /** epochs, LEARN_RATE, CONSCIENCE, NEIGHBOURHOOD_SHAPE, DIST */
    def learn(randoms : (Double, Double), parameters : List[(Int,Double,Double,Int,Int)]) = {
        nn.randomize(randoms._1, randoms._2)
        for { (epochs, learn_rate, conscience, shape, neigh) <- parameters } yield {
            kohonenLay.LEARN_RATE = learn_rate
            kohonenLay.CONSCIENCE = conscience
            kohonenLay.neigh_dist = neigh
            kohonenLay.neigh_shape = shape
            nn.learn(inputs, epochs)
        }
        println(nn.toString)
        println
    }

    learn((0.0, 1.0), List((8000, 0.06, 1, 1, 3), (8000, 0.03, 0.5, 1, 2), (8000, 0.015, 0.25, 1, 1), (8000, 0.0075, 0.125, 1, 0)))
    printResults(nn, inputs, "Obrazki")

}
