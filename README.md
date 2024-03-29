# Medicmind Multiclass Classifier Example

Demonstrates how to use a trained multiclass tensorflow model from Medicminds AI platform (https://www.medicmind.tech). Includes a python and iPhone app demo. The neural network trained detects types of cervix. Returns type1, type2 and type3 and none of these types.

# Python Demo
 - Install Tensorflow as per instructions https://www.tensorflow.org/install/
 
 - Download https://ai.medicmind.tech/shared/models/multiclass_demo/frozen_model.pb and place under /camera/data
 
 - Execute
 ```
 python infer.py --checkpoint_dir="camera/data/frozen_model.pb" --filename="cervix.jpg"
 ```
 To use your own frozen model from Medicmind just replace the frozen_model.pb file in /camera/data and change the infer.py line
```
 classes=['None','Type 1','Type 2','Type 3']
```

 to the classes you have defined in your multiclass classifier. So if your classes are dogs, cats and mice then line would become
 ```
classes = ['dog','cat','mouse']
```

 
# IOS demo
To create an IOS app using a trained frozen_model.pb download Tensorflow examples:

https://github.com/tensorflow/examples

Then convert the frozen_model.pb file from your Medicmind model to Keras TFLite format using 
```
pbtotflite.py 
```
Place the frozen_model.pb in camera/data and this python code will generate pruned.lite

Then with the IOS image classification app in Tensorflow examples under:

```
lite/examples/image_classification/ios/ImageClassification
```

Replace the 'mobilenet_quant_v1_224.tflite' with the 'pruned.lite' model under the folder 'Model' 

And also change the reference in 'ModelDataHandler.swift' to pruned.lite:

```
static let modelInfo: FileInfo = (name: "mobilenet_quant_v1_224", extension: "tflite")
```

Also included is the legacy IOS demo for using Medicmind models with IOS:
## Installation
 - Clone Tensorflow onto your Mac (Tensorflow 1.2.0 will work fine) 

 - Place the multiclass_demo code under tensorflow/tensorflow/examples

 - Download the cervix model https://ai.medicmind.tech/shared/models/multiclass_demo/frozen_model.pb  or use your own Medicmind model

 - Place frozen_model.pb under /tensorflow/examples/multiclass_demo/camera/data

## Running the Samples using CocoaPod
 - You'll need Xcode 7.3 or later.

 - Change directory to one of the samples, download the TensorFlow-experimental
   pod, and open the Xcode workspace. Observe: installing the pod can take a
   long time since it is big (~450MB). For example, if you want to run the
   simple example, then:
```bash
cd tensorflow/examples/multiclass_demo/camera
pod install
open tf_camera_example.xcworkspace # obs, not the .xcodeproj directory
```

### Troubleshooting

 - Make sure you use the TensorFlow-experimental pod (and not TensorFlow).

 - The TensorFlow-experimental pod is current about ~450MB. The reason it is
   so big is because we are bundling multiple platforms, and the pod includes
   all TensorFlow functionality (e.g. operations). The final app size after
   build is substantially smaller though (~25MB). Working with the complete
   pod is convenient during development, but see below section on how you can
   build your own custom TensorFlow library to reduce the size.

### Creating Your own App

 - Create your own app using Xcode then add a file named Podfile at the project
   root directory with the following content:
```bash
target 'YourProjectName'
       pod 'TensorFlow-experimental'
```

 - Then you run ```pod install``` to download and install the
 TensorFlow-experimental pod, and finally perform
 ```open YourProjectName.xcworkspace``` and add your code.

 - In your apps "Build Settings", make sure to add $(inherited) to sections
   "Other Linker Flags", and "Header Search Paths".

 - That's it. If you want to create your custom TensorFlow iOS library, for
   example to reduce binary footprint, see below section.

## Building the TensorFlow iOS libraries from source

 - You'll need Xcode 7.3 or later, with the command-line tools installed.

 - Follow the instructions at
   [tensorflow/contrib/makefile](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile)
   under "iOS" to compile a static library containing the core TensorFlow code.

 - You should see a single-screen app with a "Run Model" button. Tap that, and
   you should see some debug output appear below indicating that the example
   Grace Hopper image has been analyzed, with a military uniform recognized.

 - Once you have success there, make sure you have a real device connected and
   open up the Xcode project in the `camera` subfolder. Once you build and run
   that, you should get a live camera view that you can point at objects to get
   real-time recognition results.

### Troubleshooting

If you're hitting problems, here's a checklist of common things to investigate:

 - Make sure that you've run the `build_all_ios.sh` script.
   This will run `download_dependencies.sh`,`compile_ios_protobuf.sh` and `compile_ios_tensorflow.sh`.
   (check each one if they have run successful.)

 - Check that you have version 7.3 of Xcode.

 - If there's a complaint about no Sessions registered, that means that the C++
   global constructors that TensorFlow relies on for registration haven't been
   linked in properly. You'll have to make sure your project uses force_load, as
   described below.

### Creating your Own App from your source libraries

You'll need to update various settings in your app to link against
TensorFlow. You can view them in the example projects, but here's a full
rundown:

 - The `compile_ios_tensorflow.sh` script builds a universal static library in
   `tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a`. You'll need to add
   this to your linking build stage, and in Search Paths add
   `tensorflow/contrib/makefile/gen/lib` to the Library Search Paths setting.

 - You'll also need to add `libprotobuf.a` and `libprotobuf-lite.a` from
   `tensorflow/contrib/makefile/gen/protobuf_ios/lib` to your _Build Stages_ and
   _Library Search Paths_.

 - The _Header Search_ paths needs to contain:
   - the root folder of tensorflow,
   - `tensorflow/contrib/makefile/downloads/protobuf/src`
   - `tensorflow/contrib/makefile/downloads`,
   - `tensorflow/contrib/makefile/downloads/eigen`, and
   - `tensorflow/contrib/makefile/gen/proto`.

 - In the Linking section, you need to add `-force_load` followed by the path to
   the TensorFlow static library in the _Other Linker_ Flags section. This ensures
   that the global C++ objects that are used to register important classes
   inside the library are not stripped out. To the linker, they can appear
   unused because no other code references the variables, but in fact their
   constructors have the important side effect of registering the class.

 - You'll need to include the Accelerate framework in the "Link Binary with
   Libraries" build phase of your project.

 - C++11 support (or later) should be enabled by setting `C++ Language Dialect` to
   `GNU++11` (or `GNU++14`), and `C++ Standard Library` to `libc++`.

 - The library doesn't currently support bitcode, so you'll need to disable that
   in your project settings.

 - Remove any use of the `-all_load` flag in your project. The protocol buffers
   libraries (full and lite versions) contain duplicate symbols, and the
   `-all_load` flag will cause these duplicates to become link errors. If you
   were using `-all_load` to avoid issues with Objective-C categories in static
   libraries, you may be able to replace it with the `-ObjC` flag.

### Reducing the binary size

TensorFlow is a comparatively large library for a mobile device, so it will
increase the size of your app. Currently on iOS we see around a 11 MB binary
footprint per CPU architecture, though we're actively working on reducing that.
It can be tricky to set up the right configuration in your own app to keep the
size minimized, so if you do run into this issue we recommend you start by
looking at the simple example to examine its size. Here's how you do that:

 - Open the Xcode project in tensorflow/examples/ios/simple.

 - Make sure you've followed the steps above to get the data files.

 - Choose "Generic iOS Device" as the build configuration.

 - Select Product->Build.

 - Once the build's complete, open the Report Navigator and select the logs.

 - Near the bottom, you'll see a line saying "Touch tf_simple_example.app".

 - Expand that line using the icon on the right, and copy the first argument to
   the Touch command.

 - Go to the terminal, type `ls -lah ` and then paste the path you copied.

 - For example it might look like `ls -lah /Users/petewarden/Library/Developer/Xcode/DerivedData/tf_simple_example-etdbksqytcnzeyfgdwiihzkqpxwr/Build/Products/Debug-iphoneos/tf_simple_example.app`

 - Running this command will show the size of the executable as the
   `tf_simple_example` line.

Right now you'll see a size of around 25 MB, since it's including two
architectures (armv7 and arm64). As a first step, you should make sure the size
increase you see in your own app is similar, and if it's larger, look at the
"Other Linker Flags" used in the Simple Xcode project settings to strip the
executable.

After that, you can manually look at modifying the list of kernels
included in tensorflow/contrib/makefile/tf_op_files.txt to reduce the number of
implementations to the ones you're actually using in your own model. We're
hoping to automate this step in the future, but for now manually removing them
is the best approach.
