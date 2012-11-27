package neuralnetwork

object NeuralNetwork {
    //List of weights under given neuron
    type NeuronWeight = Double
    type NeuronWeights = List[NeuronWeight]
    type WeightMatrix = List[NeuronWeights]
    type Weights = List[Layer]

    abstract class Layer (var layer: List[NeuronWeights])
    case class BiasLayer (layerz: List[NeuronWeights]) extends Layer (layerz)
    case class NoBiasLayer (layerz: List[NeuronWeights]) extends Layer(layerz)

    trait WeightsPrinter {
        def weightsToString (weights: Weights): String =
            (for (layer <- weights) yield
                layerToString(layer.layer) + "\n").mkString

        def layerToString(layer: WeightMatrix): String =
            (for (neuronWeights <- layer) yield
                neuronWeightsToString(neuronWeights) + "| ").mkString

        def neuronWeightsToString(neuronWeights: NeuronWeights): String =
            (for (weight <- neuronWeights) yield
                weight.toString + " ").mkString
    }

    class Neuron(val activationFunction: Double => Double) {
        def calculate(x: Double): Double = activationFunction(x)
    }

    class NeuralNetwork (val weights: Weights, activationFunction: Double => Double) extends WeightsPrinter {
        def checkWeights(weights: Weights): Boolean =
            weights match {
                case Nil => {
                    //this shouldn't happen because it means empty argument list
                    false
                }
                case layer :: Nil => {
                    //nothing to do here, last layer determines only input length
                    true
                }
                case layer :: lowerLayers => {
                    val lowerLayerNeuronsCount = lowerLayers.head.layer.length
                    val isAllRight = (
                        for (neuronWeights <- layer.layer) yield neuronWeights.length == (lowerLayerNeuronsCount + 1)
                    ).forall((bool) => bool)
                    isAllRight && checkWeights(lowerLayers)
                }
            }
        assert(checkWeights(weights))
        val bias = -1.0
        val neuron = new Neuron(activationFunction)
        def scalarProduct(l1: List[Double], l2: List[Double]) = {
            if (l1.length == l2.length) (for {(x, y) <- l1 zip l2} yield x * y).sum
            else throw new Exception("");
        }
        def calculate(input: List[Double]) = {
            calculate0(input, weights)
        }
        def calculate0(input: List[Double], weights: Weights): List[Double] = {
            weights match {
                case inputLayer :: Nil =>
                    calculate1(input, inputLayer)
                case currentLayer :: lowerLayers => {
                    val precomputed = calculate0(input, lowerLayers)
                    calculate1(precomputed, currentLayer)
                }
                case Nil => throw new IllegalArgumentException("Weight list cannot be empty")
            }
        }
        def calculate1(input : List[Double], layerInput : Layer): List[Double] = {
            for (neuronWeights <- layerInput.layer) yield
                layerInput match {
                    case BiasLayer(lay) => neuron.calculate(scalarProduct(neuronWeights, bias :: input))
                    case NoBiasLayer(lay) => neuron.calculate(scalarProduct(neuronWeights, input))
                }
        }

        override def toString =
            weightsToString(weights)

	/** Randomize network's layers */
        def randomize(): Weights = {
            val random = new scala.util.Random();
            for (i <- 0 until weights.length)
                weights(i).layer = (for (nW <- weights(i).layer) yield
                                       (for (w <- nW) yield
                                           random.nextDouble()).toList).toList
            weights            
        }

    }

    class RandomWeightsGenerator {
       val random = new scala.util.Random()
       def randomLayer(inputSize : Integer, outputSize : Integer) : List[NeuronWeights] = {
           (for (i <- 0 until outputSize) yield
               (for (j <- 0 until inputSize) yield
                   random.nextDouble()).toList).toList
       }

       def randomLayers(sizes : List[Integer], activationFunction: Double => Double) : List[Layer] = {
           (for ( i <- 0 until (sizes.length - 1)) yield
              new BiasLayer( randomLayer(sizes(i+1) + 1, sizes(i)))).toList
       }
    }
}
