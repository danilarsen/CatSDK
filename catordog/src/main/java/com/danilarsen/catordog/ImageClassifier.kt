package com.danilarsen.catordog

import android.content.Context
import android.graphics.Bitmap
import com.danilarsen.catordog.ml.Model
import com.danilarsen.catordog.model.ClassifierResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class ImageClassifier(context: Context) {

    companion object {
        // Define the input size and the channels as per your model requirements
        private const val MODEL_INPUT_SIZE = 224
        private const val MODEL_INPUT_CHANNELS = 3
        // Assign the necessary pre-processing and post-processing operations
        private val postprocessNormalizeOp = NormalizeOp(0.0f, 1.0f) // Adjust as needed
    }

    // Instance of the TensorFlow Lite model
    private val model: Model = Model.newInstance(context)

    // Processor for pre-processing the image data
    private val imageProcessor: ImageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(0.0f, 255.0f))
        .build()

    // List of labels for the model's outputs, loaded from a file
    private val labels: List<String> = context.assets.open("labels.txt").bufferedReader().readLines()

    fun classifyImage(bitmap: Bitmap, callback: (ClassifierResult) -> Unit) {
        // Preprocess the image to match the model input
        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // Create TensorBuffer for model input
        val inputFeature = TensorBuffer.createFixedSize(
            intArrayOf(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, MODEL_INPUT_CHANNELS),
            DataType.FLOAT32
        )

        // Load the TensorBuffer with the processed image
        inputFeature.loadBuffer(tensorImage.buffer)

        // Run model inference and get the output
        val outputs = model.process(inputFeature)
        val outputFeature = outputs.outputFeature0AsTensorBuffer

        // Process and normalize model output if necessary
        val outputProcessor = TensorProcessor.Builder().add(postprocessNormalizeOp).build()
        val processedOutputFeature = outputProcessor.process(outputFeature)

        // Interpret the model output and obtain the index of the class with the highest probability
        val classificationResult = interpretResult(processedOutputFeature.floatArray)
        callback(classificationResult)
    }

    private fun getMax(arr: FloatArray): Int {
        var max = 0
        for (i in arr.indices) {
            if (arr[i] > arr[max]) max = i
        }
        return max
    }

    private fun interpretResult(resultArray: FloatArray, confidenceThreshold: Float = 0.5f): ClassifierResult {
        val maxIdx = getMax(resultArray)
        val confidence = resultArray[maxIdx]
        // If the confidence is less than the threshold, it returns "Unknown" or some other label by default.
        if (confidence < confidenceThreshold) {
            return ClassifierResult("Unknown", confidence)
        }
        val label = if (labels.isNotEmpty() && maxIdx < labels.size) labels[maxIdx] else "Unknown"
        return ClassifierResult(label, confidence)
    }

    // Function to release resources used by the model
    fun close() {
        model.close()
    }
}