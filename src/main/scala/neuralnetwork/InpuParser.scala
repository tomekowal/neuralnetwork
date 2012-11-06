package neuralnetwork

import neuralnetwork.NeuralNetwork._
import io.Source.fromFile

class InputParser(val file: String) {
    def inputs(string: String): List[List[Double]] =
        (for (line <- string.split('\n') if line != "") yield
            (for (input <- line.split(' ')) yield input.toDouble).toList).toList

    val source = fromFile(file)
    val string = source.mkString
    source.close()
    def inputsFromFile(): List[List[Double]] = {
        inputs(string)
    }
}