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
        val neighLayer = new KohonenLayer(List(List(0.0, 0.0, 0.0, 0.0), List(0.0, 0.0, 0.0, 0.0)))
        val neighLayer2D = new KohonenLayer(List(List(0.0, 0.0, 0.0, 0.0), List(0.0, 0.0, 0.0, 0.0), List(0.0, 0.0, 0.0, 0.0), List(0.0, 0.0, 0.0, 0.0)))

        val nn = new NeuralNetwork(List(kohLayer))
        val nn1i1o = new NeuralNetwork(List(new KohonenLayer(List(neuronWeights))))
        val nn4i2o = new NeuralNetwork(List(neighLayer))
        val nn4i4o = new NeuralNetwork(List(neighLayer2D))
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

   test("Teaching one input one output network eventually produces : weight = training_sample") {
        new TestNetworks {
            nn1i1o.randomize()
            nn1i1o.learn(List(neuronWeights), 300)
            assert(nn1i1o.calculate(neuronWeights) === List(1.0))
            assert(error(nn1i1o.weights(0).layer(0), neuronWeights) < epsilon)
        }
   }

   test("Teaching with 1D neighbourhood") {
        new TestNetworks {
            nn4i2o.randomize()

            neighLayer.neighbourhood_shape = 1
            neighLayer.neighbourhood_dist = 1
            val nghb: NeuronWeights = List(0.0, 0.5, 0.5, 0.5)
            val nghb1: NeuronWeights = List(0.5, 0.5, 0.5, 0.0)

            nn4i2o.learn(List(nghb, nghb1), 16000)
            assert(nn4i2o.calculate(nghb) != nn4i2o.calculate(nghb1), nn4i2o.calculate(nghb) + " compare to " + nn4i2o.calculate(nghb1) + ":" + nn4i2o.toString)
        }
   }

   test("Teaching with 2D neighbourhood") {
        new TestNetworks {
            nn4i4o.randomize()

            neighLayer2D.neighbourhood_shape = 2
            neighLayer2D.neighbourhood_dist = 1
            val nghb: NeuronWeights = List(0.3, 0.5, 0.5, 0.5)
            val nghb1: NeuronWeights = List(0.5, 0.5, 0.5, 0.3)
            val nghb2: NeuronWeights = List(0.5, 0.3, 0.5, 0.5)
            val nghb3: NeuronWeights = List(0.5, 0.5, 0.5, 0.5)

            nn4i4o.learn(List(nghb, nghb1, nghb2, nghb3), 16000)
            assert(nn4i4o.calculate(nghb) != nn4i4o.calculate(nghb1), nn4i4o.calculate(nghb) + " compare to " + nn4i4o.calculate(nghb1) + ":" + nn4i4o.toString)
            assert(nn4i4o.calculate(nghb) != nn4i4o.calculate(nghb2), nn4i4o.calculate(nghb) + " compare to " + nn4i4o.calculate(nghb2) + ":" + nn4i4o.toString)
            assert(nn4i4o.calculate(nghb) != nn4i4o.calculate(nghb3), nn4i4o.calculate(nghb) + " compare to " + nn4i4o.calculate(nghb3) + ":" + nn4i4o.toString)
        }
   }

}
