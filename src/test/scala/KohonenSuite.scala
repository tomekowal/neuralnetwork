import neuralnetwork.NeuralNetwork._
import neuralnetwork._
import org.scalatest.FunSuite

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class KohonenSuite extends FunSuite {
    trait TestNetworks extends StringParserNeuralNetwork {
        def error(result: List[Double], exact: List[Double]): Double =
            (for ( (x, y) <- result zip exact) yield (x - y)*(x - y)).sum
        def error(result: Double, exact: Double): Double = (result - exact)*(result - exact)

        val epsilon = 0.0000001

        //neuron weights
        val neuronWeights: NeuronWeights = List(0.0, 0.1, 0.2)
        val neuronWeights1: NeuronWeights = List(0.4, 0.4, 0.4)
        
        //layer definitions
        val kohLayer = new KohonenLayer(List(neuronWeights, neuronWeights1), false)

        val nn = new NeuralNetwork(List(kohLayer))
    }

    test("PSP for Kohonen network is based on Euclidian distance squared") {
        new TestNetworks {
            assert(error(nn.weights(0).psp(neuronWeights, neuronWeights), 0.0) < epsilon)
            assert(error(nn.weights(0).psp(neuronWeights, List(0.1, 0.1, 0.1)), 0.02) < epsilon)
        }
    }

   test("Winning neuron returns 1.0 and others return 0.0") {
        new TestNetworks {
            assert(nn.calculate(neuronWeights) === List(1.0, 0.0))
            assert(nn.calculate(neuronWeights1) === List(0.0, 1.0))
            val mid = (for ( (x, y) <- neuronWeights zip neuronWeights1) yield (x + y)/2.0).toList
            assert(nn.calculate( mid ) === List(1.0, 0.0))
        }
   }
}
