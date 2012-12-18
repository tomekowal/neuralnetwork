package neuralnetwork

import neuralnetwork.NeuralNetwork._
import neuralnetwork._

abstract class Teacher(val learnRate: Double) {
    def teach(layer: Layer, examples: List[(List[Double], List[Double])]) = null

    def psp(weights: NeuronWeights, inputs: List[Double]) = {
        (for {(x, y) <- weights zip inputs} yield x * y).sum
    }
    def numericDerivative(function: Double => Double, x: Double) = {
        val epsilon = 0.0001
        (function(x+epsilon) - function(x-epsilon)) / (2 * epsilon)
    }
}

class DeltaRuleTeacher(val learnRatey: Double) extends Teacher(learnRatey) {
    override def teach(layer: Layer, examples: List[(List[Double], List[Double])]) = {
        //iterate over training examples
        for {(input, targetOutput) <- examples} yield {
            val layerOutput = layer.calculate(input)
            assert(layerOutput.length == targetOutput.length && targetOutput.length == layer.layer.length)
            //crate new weights for every neuron
            val newLayer = for {(targetValue, layerValue, neuronWeights) <- (targetOutput, layerOutput, layer.layer).zipped.toList} yield {
                val weightedSum = psp(neuronWeights, input)
                val derivative = numericDerivative(layer.activationFunction, weightedSum)
                //create weights for single neuron
                val newWeights = for {(inputValue, weight) <- input zip neuronWeights} yield {
                    val delta = learnRate * (targetValue - layerValue) * derivative * inputValue
                    weight + delta
                }
                newWeights
            }
            layer.layer = newLayer
        }
        null
    }
}

class WidrowHoffTeacher(val learnRatex: Double) extends DeltaRuleTeacher(learnRatex) {
    override def numericDerivative(function: Double => Double, x: Double) = 1
}