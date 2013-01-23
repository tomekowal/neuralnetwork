package neuralnetwork

import neuralnetwork.NeuralNetwork._
import neuralnetwork._

class BackPropagationTeacher(val learnRate: Double, val momentum: Double, val iterations: Int) {
    def teach(nn: NeuralNetwork,
                       examples: List[(List[Double], List[Double])]) {
        //iterate over training examples
        for (i <- 1 to iterations) {
            for {(input, targetOutput) <- examples} yield {
                val networkOutput = nn.calculate(input)
                nn.backPropagate(networkOutput, targetOutput, learnRate, momentum)
            }
        }
    }

    def get_error(nn : NeuralNetwork, examples : List[(List[Double], List[Double])]) = {
         (for {(input,output) <- examples} yield {
	    math.sqrt(math.pow((for {(o1,o2) <- nn.calculate(input) zip output} yield {(o1-o2)*(o1-o2) }).sum,
                               2.0) / output.length)
		     
	 }).sum
    }
}
