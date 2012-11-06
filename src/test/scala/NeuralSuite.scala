import neuralnetwork.NeuralNetwork._
import neuralnetwork._
import org.scalatest.FunSuite

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class NeuralSuite extends FunSuite {
    trait TestNetworks extends StringParserNeuralNetwork {
        def error(result: List[Double], exact: List[Double]): Double =
            (for ( (x, y) <- result zip exact) yield (x - y)*(x - y)).sum

        val epsilon = 0.0000001

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

        //test inputs
        val input = List(0.0, 1.0, 2.0)

        val weightsString =
            """0.0 0.1 0.2 | 0.0 0.1 0.2
              |0.0 0.1 0.2 | 0.0 0.1 0.2""".stripMargin
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

    test("string parser") {
        new TestNetworks {
            object TestParser extends StringParserNeuralNetwork {
                assert(weights(weightsString) === List(layerWithTwoNeurons, layerWithTwoNeurons))
            }
        }
    }

    test("file parser") {
        new TestNetworks {
            val parser = new FileParserNeuralNetwork("src/test/resources/weights.txt")
            assert(parser.weightsFromFile === List(layerWithTwoNeurons, layerWithTwoNeurons))
        }
    }

    test("return zero when weights are zero") {
        new TestNetworks {
            override val weightsString =
                """0.0 0.0 0.0 | 0.0 0.0 0.0
                  |0.0 0.0 0.0 | 0.0 0.0 0.0""".stripMargin
            val w = weights(weightsString)
            val fa = (x: Double) => x
            val nn = new NeuralNetwork(w, fa)
            assert(nn.calculate(List(1.0, 1.0)) === List(0.0, 0.0))
        }
    }

    test("test one layer network") {
        new TestNetworks {
            override val weightsString = "0.5 0.4 0.3 | 0.2 0.1 0.0"
            val w = weights(weightsString)
            val fa = (x: Double) => x
            val nn = new NeuralNetwork(w, fa)
            val result = nn.calculate(List(1.0, 1.0))
            val exact = List(1.2, 0.3)
            assert(error(result, exact) < epsilon)
        }
    }

    test("test non zero values") {
        new TestNetworks {
            override val weightsString =
                """0.3 0.2 0.1 | 0.3 0.2 0.1
                  |0.1 0.2 0.3 | 0.1 0.2 0.3""".stripMargin
            val w = weights(weightsString)
            val fa = (x: Double) => x
            val nn = new NeuralNetwork(w, fa)
            val result = nn.calculate(List(1.0, 1.0))
            val exact = List(0.48, 0.48)
            assert(error(result, exact) < epsilon)
        }
    }

    test("test check weights") {
        new TestNetworks {
            override val weightsString =
                """0.3 0.2 0.1 | 0.3 0.2 0.1
                  |0.1 0.2 0.3""".stripMargin
            val w = weights(weightsString)
            val fa = (x: Double) => x
            intercept[java.lang.AssertionError] {
                val nn = new NeuralNetwork(w, fa)
            }
        }
    }

    test("parse input") {
        new TestNetworks {
            val inputParser = new InputParser("src/test/resources/inputs.txt")
            assert(inputParser.inputsFromFile === List(List(1.0, 0.0), List(0.5, 0.5)))
        }
    }
}
