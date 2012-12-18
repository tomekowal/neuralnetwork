package neuralnetwork

import scala.math._

object NeuralNetwork {
    //List of weights under given neuron
    type NeuronWeight = Double
    type NeuronWeights = List[NeuronWeight]
    type WeightMatrix = List[NeuronWeights]
    type Weights = List[Layer]

    abstract class Layer (var layer: List[NeuronWeights], val bias: Boolean) {
         def activationFunction(x: Double): Double

         def calculate(inps: List[Double]) = {
                var inputs : List[Double] = null
                if (bias) inputs = -1.0 :: inps else inputs = inps

            for (weights <- layer) yield {
                if (weights.length != inputs.length)
                    throw new Exception("weights and inputs not of equal length...")
                activationFunction(psp(weights, inputs))
            }
	 }

         def psp(weights: NeuronWeights, inputs: List[Double]) = {
            (for {(x, y) <- weights zip inputs} yield x * y).sum
        }

        def replace_weights(which : Int, withWhat : NeuronWeights) = {
            replace_weights1(which, withWhat, layer, List())
        }

        def replace_weights1(which : Int, withWhat : NeuronWeights, weights : List[NeuronWeights], acu : List[NeuronWeights]): List[NeuronWeights] = {
            val current :: nextNeurons = weights
            if (which == 0)
                acu.reverse ++ (withWhat :: nextNeurons)
            else
                replace_weights1(which-1, withWhat, nextNeurons, current::acu)
        }

        def learn(trainingSet : List[List[Double]]) : List[NeuronWeights] = null
    }
    case class LinearLayer (layerx: List[NeuronWeights], biasx: Boolean = true) extends Layer (layerx, biasx) {
         override def activationFunction(x: Double): Double = x
    }
    case class SigmoidLayer (layerx: List[NeuronWeights], biasx: Boolean = true) extends Layer (layerx, biasx) {
         override def activationFunction(x: Double): Double = 1.0 / (1.0 + exp(-x))
    }
    case class ThresholdLayer (layerx: List[NeuronWeights], biasx: Boolean = true) extends Layer (layerx, biasx) {
         override def activationFunction(x: Double): Double = if (x > 0) 1.0 else 0.0
    }
    case class KohonenLayer (layerx: List[NeuronWeights]) extends Layer (layerx, false) {
         def this(inputs : Int, outputs : Int) = this( (for {i <- 0 to outputs-1} yield Array.fill(inputs)(0.0).toList).toList )

        var LEARN_RATE : Double = 0.03
        var CONSCIENCE : Double = 1.0
        var neigh_shape : Int = 1
        var neigh_dist : Int= 0
        var winning_count = Array.fill(layer.length)(0)

        override def activationFunction(x: Double): Double = x
        override def psp(weights: NeuronWeights, inputs: List[Double]) = {
            (for {(x, y) <- weights zip inputs} yield (x - y)*(x - y)).sum
        }

        def output_distances(inputs: List[Double]) = {
	    (for (weights <- layer) yield {
                if (weights.length != inputs.length)
                    throw new Exception("weights and inputs not of equal length...")
                psp(weights, inputs)}).toList
        }

        def mark_winner(outputs : List[Double]) = {
            val min = outputs.min
            val result = Array.fill(outputs.length) (0.0)
            result(outputs.indexOf(min)) = 1.0
            result.toList
        }

        override def calculate(inputs: List[Double]) = {
	    mark_winner(output_distances(inputs))
        }

        override def learn(trainingSet : List[List[Double]]) : List[NeuronWeights] = {
            for {trainingExample <- trainingSet} yield {
                val result = adjusted_distance(trainingExample)
                val winner = result.indexOf(1.0)
                val neurons_to_teach = get_neighbourhood(winner)
                teach(neurons_to_teach, trainingExample)
            }
            layer
        }

        def adjusted_distance(inputs: List[Double]) : List[Double] = {
            val len = layer.length
	    val result = mark_winner( (for {(output, winFreq) <- (output_distances(inputs) zip winning_count)} yield output + CONSCIENCE*(winFreq/len - 1)).toList )
            winning_count(result.indexOf(1.0)) += 1
            result
        }

        def get_neighbourhood(i : Int) : List[(Int, Int)] = {
            val min = (x : Int, y : Int) => if (x<y) x else y
            val max = (x : Int, y : Int) => if (x>y) x else y

            neigh_shape match {
                case 1 =>
                    val row_size = layer.length
                    (for { x <- -neigh_dist to neigh_dist
                          if ( x+i>= 0 && x+i< row_size ) }
                              yield (abs(x-i), x+i) ).toList

                case 2 =>
                    val row_size = sqrt(layer.length) toInt
                    val row = i / row_size
                    val col = i % row_size

                    (for { rowz <- -neigh_dist to neigh_dist; colz <- -neigh_dist to neigh_dist
                          if (rowz >= 0 && rowz < row_size && colz >= 0 && colz < row_size && abs(rowz) + abs(colz) < neigh_dist)}
                              yield (abs(rowz) + abs(colz), rowz*row_size + colz)).toList

            }
        }


        def teach(neurons : List[(Int, Int)], input : List[Double]) {
            for {(dist, neuron) <- neurons} yield
                layer = replace_weights(neuron, (for {(weight, inp) <- (layer(neuron) zip input)} yield weight + (LEARN_RATE/(dist+1)) * (inp - weight)).toList)
        }

    }

    trait WeightsPrinter {
        def weightsToString (weights: Weights): String =
            (for (layer <- weights) yield
                layerToString(layer.layer) + "\n\n").mkString

        def layerToString(layer: WeightMatrix): String =
            (for (neuronWeights <- layer) yield
                neuronWeightsToString(neuronWeights) + "\n ").mkString

        def neuronWeightsToString(neuronWeights: NeuronWeights): String =
            (for (weight <- neuronWeights) yield
                weight.toString + " ").mkString
    }

    class NeuralNetwork (val weights: Weights) extends WeightsPrinter {
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
                    val bias = if (layer.bias) 1 else 0
                    val lowerLayerNeuronsCount = lowerLayers.head.layer.length
                    println(lowerLayerNeuronsCount)
                    val isAllRight = (
                        for (neuronWeights <- layer.layer) yield neuronWeights.length == (lowerLayerNeuronsCount + bias)
                    ).forall((bool) => bool)
                    isAllRight && checkWeights(lowerLayers)
                }
            }
        assert(checkWeights(weights))

        def calculate(input: List[Double]) = {
            calculate0(input, weights)
        }
        def calculate0(input: List[Double], weights: Weights): List[Double] = {
            weights match {
                case inputLayer :: Nil =>
                    inputLayer.calculate(input)
                case currentLayer :: lowerLayers => {
                    val precomputed = calculate0(input, lowerLayers)
                    currentLayer.calculate(precomputed)
                }
                case Nil => throw new IllegalArgumentException("Weight list cannot be empty")
            }
        }

        def learn(trainingSet : List[List[Double]], epoches : Int) = {
             for (epoche <- (0 to epoches-1).toList)
                 for {layer <- weights} layer.learn(trainingSet)
        }

        override def toString =
            weightsToString(weights)

	/** Randomize network's layers */
        def randomize(a : Double = 0.0, b : Double = 1.0): Weights = {
            val random = new scala.util.Random();
            val len = b - a
            for (i <- 0 until weights.length)
                weights(i).layer = (for (nW <- weights(i).layer) yield
                                       (for (w <- nW) yield
                                           a + random.nextDouble()*len).toList).toList
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

       def randomLayers(sizes : List[Integer]) : List[Layer] = {
           (for ( i <- 0 until (sizes.length - 1)) yield
              new LinearLayer( randomLayer(sizes(i+1) + 1, sizes(i)))).toList
       }
    }
}
