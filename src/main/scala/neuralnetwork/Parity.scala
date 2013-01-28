package neuralnetwork
import neuralnetwork.NeuralNetwork._
import scala.math._

object Parity extends App {
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

    def createListsOfLength(l : Int, n : Int): List[List[Double]] = {
    	var result = List[List[Double]]()
    	for (i <- 1 to l) {
	    	result = createListLength(n) :: result
	}
	result
    }

    def createListLength(n : Int): List[Double] = {
    	var result = List[Double]()

	for (i <- 1 to n) {
	    result = 0.0 :: result
	}
	result
    }

    val hidden_layer = 3

    val nn = new NeuralNetwork(List(new SigmoidLayer(createListsOfLength(1, hidden_layer+1), true), new SigmoidLayer(createListsOfLength(hidden_layer, 4), true)))
    nn.randomize(0.0, 0.1)
    val inps = List ( List(0.0,0.0,0.0),
    		      List(0.0,0.0,1.0),
		      List(0.0,1.0,0.0),
		      List(0.0,1.0,1.0),
		      List(1.0,0.0,0.0),
		      List(1.0,0.0,1.0),
		      List(1.0,1.0,0.0),
		      List(1.0,1.0,1.0) )

    val outputs = List ( List(1.0),
       		         List(0.0),
		         List(0.0),
		         List(1.0),
			 List(0.0),
			 List(1.0),
			 List(1.0),
			 List(0.0) )
    val training = (inps zip outputs).toList

    val teacher = new BackPropagationTeacher(0.2, 0.5, 1000)
    val iterations = 20
    println("err: " + (for { i <- 1 to iterations } yield {
         teacher.teach(nn, training)
	 (i, teacher.get_error(nn, training))
	 }).toList)
	
    printResults(nn, 
		 inps,
		 "parity")
    println("Error: " + teacher.get_error(nn, training))

    println(nn.toString)



}
