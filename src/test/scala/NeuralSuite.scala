import neuralnetwork.NeuralNetwork._
import neuralnetwork.StringParserNeuralNetwork
import org.scalatest.FunSuite

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class NeuralSuite extends FunSuite {
    trait TestNetworks {
        //activation function
        val linearActivationFunction = (x: Double) => x
        //neuron weights
        val neuronWithSingleInputWeight: NeuronWeights = List(0.5)
        val neuronWeights: NeuronWeights = List(0.0, 0.1, 0.2)
        val zeroNeuronWeights: NeuronWeights = List(0.0, 0.0, 0.0)
        //layer definitions
        val layerOfZeroes = List(zeroNeuronWeights)
        val layerWithOneNeuron = List(neuronWeights)
        val layerWithTwoNeurons = List(neuronWeights, neuronWeights)
        val endLayer = List(neuronWithSingleInputWeight)
        //weights definition
        val oneLayerOfZeroes = List(layerOfZeroes)
        val oneLayerWeights = List(layerWithOneNeuron)
        val oneLayerTwoNeuronsWeights = List(layerWithTwoNeurons)
        val twoLayersWithSingleNeuron = List(endLayer, layerWithOneNeuron)

        //neural network definitions
        val nn0 = new NeuralNetwork(oneLayerOfZeroes, linearActivationFunction)
        val nn1 = new NeuralNetwork(oneLayerWeights, linearActivationFunction)
        val nn12 = new NeuralNetwork(oneLayerTwoNeuronsWeights, linearActivationFunction)
        val nn21 = new NeuralNetwork(twoLayersWithSingleNeuron, linearActivationFunction)

        //test inputs
        val input = List(0.0, 1.0, 2.0)
    }

    test("calculate neuron output") {
        val neuron = new Neuron((x: Double) => x)
        assert(neuron.calculate(2.5) === 2.5)
    }

    test("scalar product") {
        new TestNetworks {
            assert(nn1.scalarProduct(input, input) === 5)
        }
    }

    test("return 0 when weights are 0") {
        new TestNetworks {
            assert(nn0.calculate(input) === List(0))
        }
    }

    test("test non zero weigts") {
        new TestNetworks {
            assert(nn1.calculate(input) === List(0.5))
        }
    }

    test("two neurons one layer") {
        new TestNetworks {
            assert(nn12.calculate(input) === List(0.5, 0.5))
        }
    }

    test("multiple layers") {
        new TestNetworks {
            assert(nn21.calculate(input) === List(0.25))
        }
    }

    test("string parser") {
        new TestNetworks {
            object TestParser extends StringParserNeuralNetwork {
                val weightsString =
                    """0.0 0.1 0.2 | 0.1 0.2 0.3
                      |0.1 0.2 0.3 | 0.1 0.2 0.3""".stripMargin

                assert(weights(weightsString) === List(layerWithTwoNeurons, layerWithTwoNeurons))
            }
        }
    }
}
