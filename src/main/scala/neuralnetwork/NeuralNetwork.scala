package neuralnetwork

object NeuralNetwork {
    //List of weigts under given neuron
    type NeuronWeight = Double
    type NeuronWeights = List[NeuronWeight]
    type Layer = List[NeuronWeights]
    type Weights = List[Layer]

    class Neuron(val activationFunction: Double => Double) {
        def calculate(x: Double): Double = activationFunction(x)
    }

    class NeuralNetwork(val weights: Weights, activationFunction: Double => Double) {
        val neuron = new Neuron(activationFunction)
        def scalarProduct(l1: List[Double], l2: List[Double]) = (for {(x, y) <- l1 zip l2} yield x * y).sum
        def calculate(input: List[Double]) = {
            calculate0(input, weights)
        }
        def calculate0(input: List[Double], weights: Weights): List[Double] = {
            weights match {
                case inputLayer :: Nil => for (neuronWeights <- inputLayer) yield scalarProduct(neuronWeights, input)
                case currentLayer :: t => {
                    val precomputed = calculate0(input, t)
                    for (neuronWeights <- currentLayer) yield scalarProduct(neuronWeights, precomputed)
                }
                case Nil => throw new IllegalArgumentException("Weight list cannot be empty")
            }
        }
    }
}