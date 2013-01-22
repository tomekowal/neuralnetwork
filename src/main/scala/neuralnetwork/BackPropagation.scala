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
}

