package neuralnetwork

object NeuralNetwork {
    //List of weights under given neuron
    type NeuronWeight = Double
    type NeuronWeights = List[NeuronWeight]
    type Layer = List[NeuronWeights]
    type Weights = List[Layer]

    trait WeightsPrinter {
        def weightsToString (weights: Weights): String =
            (for (layer <- weights) yield
                layerToString(layer) + "\n").mkString

        def layerToString(layer: Layer): String =
            (for (neuronWeights <- layer) yield
                neuronWeightsToString(neuronWeights) + "| ").mkString

        def neuronWeightsToString(neuronWeights: NeuronWeights): String =
            (for (weight <- neuronWeights) yield
                weight.toString + " ").mkString
    }

    class Neuron(val activationFunction: Double => Double) {
        def calculate(x: Double): Double = activationFunction(x)
    }

    class NeuralNetwork(val weights: Weights, activationFunction: Double => Double) extends WeightsPrinter{
        val neuron = new Neuron(activationFunction)
        def scalarProduct(l1: List[Double], l2: List[Double]) = (for {(x, y) <- l1 zip l2} yield x * y).sum
        def calculate(input: List[Double]) = {
            calculate0(input, weights)
        }
        def calculate0(input: List[Double], weights: Weights): List[Double] = {
            weights match {
                case inputLayer :: Nil =>
                    for (neuronWeights <- inputLayer) yield
                        neuron.calculate(scalarProduct(neuronWeights, input))
                case currentLayer :: lowerLayers => {
                    val precomputed = calculate0(input, lowerLayers)
                    for (neuronWeights <- currentLayer) yield
                        neuron.calculate(scalarProduct(neuronWeights, precomputed))
                }
                case Nil => throw new IllegalArgumentException("Weight list cannot be empty")
            }
        }
        override def toString =
            weightsToString(weights)
    }
}