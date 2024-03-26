package com.danilarsen.catordog

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.danilarsen.catordog.ml.Model
import com.danilarsen.catordog.model.ClassifierResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ImageClassifier(context: Context) {

    companion object {
        private const val TAG = "ImageClassifier"
    }

    // Instance of the TensorFlow Lite model
    private val model: Model = Model.newInstance(context)

    // Processor for pre-processing the image data
    private val imageProcessor: ImageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)) // Resize the image to 224x224 pixels
        .build()

    // List of labels for the model's outputs, loaded from a file
    private val labels: List<String> = context.assets.open("labels.txt").bufferedReader().readLines()

    // Function to classify an image and return the result via a callback
    fun classifyImage(bitmap: Bitmap, callback: (ClassifierResult) -> Unit) {

        // Convert the bitmap into a TensorImage
        val tensorImage = TensorImage(DataType.FLOAT32).apply {
            load(bitmap)
            imageProcessor.process(this) // Apply image processing
        }

        // Buffer to hold model input data
        val modelInputBuffer = tensorImage.buffer

        // Creating a TensorBuffer to hold the input data
        val inputFeature0 = TensorBuffer.createFixedSize(
            intArrayOf(1, 224, 224, 3), DataType.FLOAT32
        )
        inputFeature0.loadBuffer(modelInputBuffer)

        // Process the input through the model
        val outputs = model.process(inputFeature0)

        // Retrieve the output of the model
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

        model.close()
        Log.d(TAG, "Output tensor: ${outputFeature0.joinToString(", ")}")

        // Interpret the result of the model
        val classificationResult = interpretResult(outputFeature0)

        // Invoke the callback with the classification result
        callback(classificationResult)
    }

    // Function to interpret the result from the model
    private fun interpretResult(resultArray: FloatArray): ClassifierResult {
        // Determine the class with the highest confidence score
        val predictedClassIndex = resultArray.indices.maxByOrNull { resultArray[it] } ?: -1
        Log.d(TAG, "predictedClassIndex $predictedClassIndex")

        // Retrieve the confidence of the predicted class
        val maxConfidence = if (predictedClassIndex != -1) resultArray[predictedClassIndex] else 0f
        Log.d(TAG, "maxConfidence $maxConfidence")

        // Map the predicted class index to the corresponding label
        val label = if (predictedClassIndex != -1) labels[predictedClassIndex] else "Unknown"

        // Return the classification result
        return ClassifierResult(label, maxConfidence)
    }

    // Function to release resources used by the model
    fun close() {
        model.close()
    }
}