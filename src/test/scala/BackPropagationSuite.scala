import neuralnetwork.NeuralNetwork._
import neuralnetwork._
import org.scalatest.FunSuite

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class BackPropagationSuite extends FunSuite {
    trait TestNetworks extends StringParserNeuralNetwork {
        def error(result: List[Double], exact: List[Double]): Double =
            (for ( (x, y) <- result zip exact) yield (x - y)*(x - y)).sum
        def error(result: Double, exact: Double): Double = (result - exact)*(result - exact)

        val epsilon = 0.0000001

        //neuron weights
        val neuronWeights: NeuronWeights = List(0.0, 0.1, 0.2)
        val neuronWeights1: NeuronWeights = List(0.4, 0.4, 0.4)
    }

    test("Remember input") {
        new TestNetworks {
            var layer = new LinearLayer(List(neuronWeights, neuronWeights1), false)
            val input = List(0.1, 0.2, 0.3)
            val nn1 = new NeuralNetwork(List(layer))
            nn1.calculate(input)
            assert(layer.inputs === input)
        }
    }

    test("Calculate output deltas") {
        new TestNetworks {
            var layer = new LinearLayer(List(neuronWeights, neuronWeights1), false)
            val input = List(0.0, 0.1, 0.2)
            val targetOutput = List(1.0, 2.0)
            val nn1 = new NeuralNetwork(List(layer))
            val networkOutput = nn1.calculate(input)
            nn1.calculateDeltas(networkOutput, targetOutput)
            assert(layer.inputs === input)
        }
    }

    test("calculate input deltas") {
        val singleNeuron = List(0.5)
        val oneNeuronLayer = new LinearLayer(List(singleNeuron), false)
        val secondNeuronLayer = new LinearLayer(List(singleNeuron), false)

        val input = List(1.0)
        val targetOutput = List(3.0)
        val nn1 = new NeuralNetwork(List(oneNeuronLayer, secondNeuronLayer))
        val networkOutput = nn1.calculate(input)

        nn1.calculateDeltas(networkOutput, targetOutput)
        assert(oneNeuronLayer.deltas === List(2.75))
        assert(secondNeuronLayer.deltas === List(1.375))
    }

    test("backpropagate") {
        new TestNetworks {
            val singleNeuron = List(1.0)
            val oneNeuronLayer = new LinearLayer(List(singleNeuron), false)
            val secondNeuronLayer = new LinearLayer(List(singleNeuron), false)

            val input = List(1.0)
            val targetOutput = List(3.0)
            val nn1 = new NeuralNetwork(List(oneNeuronLayer, secondNeuronLayer))
            val networkOutput = nn1.calculate(input)

            nn1.backPropagate(networkOutput, targetOutput, 0.1)
            assert(oneNeuronLayer.layer(0)(0) - 1.2 < epsilon)
            assert(secondNeuronLayer.layer(0)(0) - 1.2 < epsilon)
        }
    }

    test("bias two layer network") {
        var inputBiasLayer = new LinearLayer(List(List(0.5, 0.5)), true)
        var outputBiasLayer = new LinearLayer(List(List(0.5, 0.5)), true)
        val input = List(1.0)
        val targetOutput = List(2.0)
        var nn = new NeuralNetwork(List(outputBiasLayer, inputBiasLayer))
        val networkOutput = nn.calculate(input)

        nn.backPropagate(networkOutput, targetOutput, 0.1)
    }
}
