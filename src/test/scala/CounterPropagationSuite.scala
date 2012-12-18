import neuralnetwork.NeuralNetwork._
import neuralnetwork._
import org.scalatest.FunSuite

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class CounterPropagationSuite extends FunSuite {
    trait TestNetworks extends StringParserNeuralNetwork {
        def error(result: List[Double], exact: List[Double]): Double =
            (for ( (x, y) <- result zip exact) yield (x - y)*(x - y)).sum
        def error(result: Double, exact: Double): Double = (result - exact)*(result - exact)

        val epsilon = 0.0000001

        //neuron weights
        val neuronWeights: NeuronWeights = List(0.0, 0.1, 0.2)
        val neuronWeights1: NeuronWeights = List(0.4, 0.4, 0.4)

        //layer definitions
        val kohLayer = new KohonenLayer(List(neuronWeights, neuronWeights1))
        val neighLayer = new KohonenLayer(4, 2)
        val neighLayer2D = new KohonenLayer(4, 4)

        val nn = new NeuralNetwork(List(kohLayer))
        val nn1i1o = new NeuralNetwork(List(new KohonenLayer(List(neuronWeights))))
        val nn4i2o = new NeuralNetwork(List(neighLayer))
        val nn4i4o = new NeuralNetwork(List(neighLayer2D))
    }

    test("Delta rule one layer") {
        new TestNetworks {
            var layer = new LinearLayer(List(neuronWeights, neuronWeights1), false)
            val learnRate = 0.1
            val teacher = new DeltaRuleTeacher(learnRate)
            val input = List(0.1, 0.2, 0.3)
            val output = List(1.0, 1.0)
            teacher.teach(layer, List( (input, output) ))
            assert(error(layer.layer(0), List(0.0092, 0.1184, 0.2276)) < epsilon)
            assert(error(layer.layer(1), List(0.4076, 0.4152, 0.4228)) < epsilon)
        }
    }

    test("Widrow Hoff one layer") {
        new TestNetworks {
            var layer = new LinearLayer(List(neuronWeights, neuronWeights1), false)
            val learnRate = 0.1
            val teacher = new WidrowHoffTeacher(learnRate)
            val input = List(0.1, 0.2, 0.3)
            val output = List(1.0, 1.0)
            teacher.teach(layer, List( (input, output) ))
            assert(error(layer.layer(0), List(0.0092, 0.1184, 0.2276)) < epsilon)
            assert(error(layer.layer(1), List(0.4076, 0.4152, 0.4228)) < epsilon)
        }
    }

    test("Numeric derivative") {
        new TestNetworks {
            val learnRate = 0.1
            val teacher = new DeltaRuleTeacher(learnRate)
            val derivative = teacher.numericDerivative( (x: Double) => x*x, 1.0)
            assert(error(derivative, 2.0) < epsilon)
        }
    }
}
