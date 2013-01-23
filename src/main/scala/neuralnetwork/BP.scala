package neuralnetwork
import neuralnetwork.NeuralNetwork._
import scala.math._

object BP extends App {
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

    val weights = new FileParserNeuralNetwork("bpweights.txt").weightsFromFile
    val nn = new NeuralNetwork(weights)
    nn.randomize(0.0, 0.2)

    val training = List( ( List(1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0), List(1.0,0.0,0.0)),
    		      	    ( List(1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), List(0.0,1.0,0.0)),
			    ( List(0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0), List(0.0,0.0,1.0)) )

    val teacher = new BackPropagationTeacher(0.4, 0.3, 1)
    val iterations = 2000
    println("err: " + (for { i <- 1 to iterations } yield {
         teacher.teach(nn, training)
	 (i, teacher.get_error(nn, training))
	 }).toList)
	
    printResults(nn, 
		 List( List(1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0),
    		       List(1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),
		       List(0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0) ),
		 "hard example")
    println("Error: " + teacher.get_error(nn, training))

    println(nn.toString)



}
