# Implementing ImageClassifier in Android
## Step 1: Set Up CatSDK in Your Project
To utilize the ImageClassifier from CatSDK, start by adding CatSDK to your build.

Add the JitPack repository to your build file:

> Key Features:
Utilizes a TensorFlow Lite model for image classification.
Pre-processes images for model compatibility.
Interprets the model's output to provide classification results.

> How to get a Git project into your build:
Add the JitPack repository to your build file

```gradle
dependencyResolutionManagement {
		repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
		repositories {
			mavenCentral()
			maven(url = "https://jitpack.io")
		}
	}
```
## Step 2. Add the dependency
```gradle
dependencies {
	        implementation 'com.github.danilarsen:CatSDK:1.0.2'
	}
```

## Step 3: Using ImageClassifier in Your Application
Once CatSDK is set up, you can implement ImageClassifier as follows:
### Initialize ImageClassifier:
Create an instance of ImageClassifier in your activity or fragment.


```gradle
private lateinit var imageClassifier: ImageClassifier

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)
    imageClassifier = ImageClassifier(context)
}
```

## Classify an Image:
Use classifyImage to process an image and get classification results.

```gradle
imageClassifier.classifyImage(bitmap) { result ->
    // Handle the result (result is a ClassifierResult object)
}
```

## Resource Management:
- Don't forget to release resources when they're no longer needed by calling close.

```gradle
override fun onDestroy() {
    super.onDestroy()
    imageClassifier.close()
}
```
ImageClassifier Implementation Details:
The ImageClassifier class internally handles the following tasks:

- Model Instantiation: Loads the TensorFlow Lite model.
- Image Pre-processing: Processes input images to make them compatible with the model (e.g., resizing).
- Model Execution: Runs the model with the processed image.
- Result Interpretation: Interprets the model's output to derive classification results.

## Important Note:

We apologize for the inconvenience, but in the current version of CatSDK, you will need to explicitly include the following TensorFlow Lite dependencies in your project's build.gradle file:

```gradle
dependencies {
    implementation("org.tensorflow:tensorflow-lite-support:0.3.1")
    implementation("org.tensorflow:tensorflow-lite-metadata:0.1.0")
}
```

Our team is actively working on enhancing CatSDK to eliminate this requirement in future releases. Thank you for your patience and understanding. Please look forward to upcoming versions where adding these dependencies manually will no longer be necessary.
