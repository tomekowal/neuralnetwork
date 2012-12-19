package neuralnetwork
import neuralnetwork.NeuralNetwork._
import scala.math._

object AppCounterPropagation extends App {
    val inputs = new InputParser("grossberg.txt").inputsFromFile
    val randomLayerGenerator = new RandomWeightsGenerator

    var kohonenLay = new KohonenLayer(9, 9)
    var grossbergLay = new LinearLayer(randomLayerGenerator.randomLayer(9, 3), false)
    var nn = new NeuralNetwork(List(grossbergLay, kohonenLay))

    def fmt(v: Any): String = v match {
        case d : Double => "%1.5f" format d
        case i : Int => i.toString
        case _ => throw new IllegalArgumentException
    }

    def printList(list: List[Double]) =
        (for (elem <- list) yield fmt(elem) + ", ").mkString

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
    def learnKoh(inputs : List[List[Double]], randoms : (Double, Double), parameters : List[(Int,Double,Double,Int,Int)]) = {
        nn.randomize(randoms._1, randoms._2)
        for { (epochs, learn_rate, conscience, shape, neigh) <- parameters } yield {
            kohonenLay.LEARN_RATE = learn_rate
            kohonenLay.CONSCIENCE = conscience
            kohonenLay.neigh_dist = neigh
            kohonenLay.neigh_shape = shape
            nn.learn(inputs, epochs)

        }
        println("NETWORK KOH")
        println(nn.toString)
    }

    def learnGrossberg(learnRate: Double, epochs: Int) {
        val teacher = new WidrowHoffTeacher(learnRate)
        val firstClass = List(1.0, 0.0, 0.0)
        val secondClass = List(0.0, 1.0, 0.0)
        val thirdClass = List(0.0, 0.0, 1.0)
        val outputs = List(firstClass, firstClass, firstClass, secondClass, secondClass, secondClass, thirdClass, thirdClass, thirdClass)
        val grossInputs = for {input <- inputs} yield kohonenLay.calculate(input)

        for {iter <- 1 to epochs} {
            teacher.teach(grossbergLay, grossInputs zip outputs)
        }
        println("NETWORK GROSS")
        println(nn.toString)
    }

    println("WAGI SIECI")
    println(nn.toString)
    learnKoh(inputs, (0.0, 1.0), List((8000, 0.06, 1.0, 1, 3), (8000, 0.03, 0.5, 1, 2), (8000, 0.015, 0.25, 1, 1), (8000, 0.0075, 0.125, 1, 0)))
    learnGrossberg(0.1, 5000)
    printResults(nn, inputs, "Obrazki")
}
