package com.example.catsdk.view.fragments

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.provider.Settings
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.danilarsen.catordog.model.ClassifierResult
import com.danilarsen.catordog.ImageClassifier
import com.example.catsdk.databinding.FragmentCameraBinding


class CameraFragment : Fragment() {

    companion object {
        private const val TAG = "CameraFragment"
    }

    private lateinit var binding: FragmentCameraBinding
    private lateinit var imageClassifier: ImageClassifier

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                launchCamera()
            } else {
                showPermissionDeniedDialog()
            }
        }

    private val getContent =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                // Get the thumbnail of the extra 'data'
                val imageBitmap = result.data?.extras?.get("data") as Bitmap?
                imageBitmap?.let { bmp ->
                    // Assuming you need to rotate the image 90 degrees to the right
                    val matrix = Matrix()
                    matrix.postRotate(90f)
                    val rotatedBitmap =
                        Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)

                    binding.imageview.setImageBitmap(rotatedBitmap)
                    imageClassifier.classifyImage(rotatedBitmap) { classificationResult ->
                        updateUIWithClassification(classificationResult)
                    }

                }
            }
        }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        binding = FragmentCameraBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        imageClassifier = ImageClassifier(requireContext())
        setupListeners()
    }

    private fun setupListeners() {
        binding.takePictureButton.setOnClickListener {
            checkCameraPermissionAndTakePicture()
        }
    }

    private fun updateUIWithClassification(classificationResult: ClassifierResult) {
        // Make sure you update the UI in the main thread
        Log.i(TAG, "${classificationResult.label} - ${classificationResult.confidence}")
        binding.resultTextview.text = "${classificationResult.label} - ${classificationResult.confidence}"
    }

    private fun launchCamera() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        try {
            getContent.launch(takePictureIntent)
        } catch (e: ActivityNotFoundException) {
            // Show error if a camera app is not found
            Log.e(TAG, e.message.toString())
        }
    }

    private fun checkCameraPermissionAndTakePicture() {
        when {
            ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED -> {
                launchCamera()
            }
            shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {
                showPermissionExplanationDialog()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun showPermissionExplanationDialog() {
        AlertDialog.Builder(requireContext())
            .setTitle("Permission Explanation")
            .setMessage("We need access to your camera to take pictures. This permission is required for core functionalities of the app.")
            .setPositiveButton("OK") { _, _ ->
                // Request permission again
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showPermissionDeniedDialog() {
        // Create and display an explanatory dialog or a Snackbar, for example:
        AlertDialog.Builder(requireContext())
            .setTitle("Camera Permission Required")
            .setMessage("This functionality requires access to the camera. Please enable the permission in the app settings.")
            .setPositiveButton("Settings") { _, _ ->
                // Redirect the user to the application settings
                openAppSettings()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun openAppSettings() {
        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
        val uri = Uri.fromParts("package", requireContext().packageName, null)
        intent.data = uri
        startActivity(intent)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        //Make sure you clean the image classifier
        imageClassifier.close()
    }
}