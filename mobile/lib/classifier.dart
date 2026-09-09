import 'dart:typed_data';

import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

class Prediction {
  const Prediction({required this.label, required this.confidence});

  final String label;
  final double confidence;
}

class MaizeClassifier {
  MaizeClassifier._(this._interpreter);

  final Interpreter _interpreter;
  static const labels = ['Bad', 'Good'];

  static Future<MaizeClassifier> load() async {
    final interpreter = await Interpreter.fromAsset('assets/models/maize_classifier.tflite');
    return MaizeClassifier._(interpreter);
  }

  Prediction classify(Uint8List bytes) {
    final decoded = image_lib.decodeImage(bytes);
    if (decoded == null) {
      throw const FormatException('Unable to decode image.');
    }

    final resized = image_lib.copyResize(decoded, width: 224, height: 224);
    final inputTensor = _interpreter.getInputTensor(0);
    final inputScale = inputTensor.params.scale;
    final inputZeroPoint = inputTensor.params.zeroPoint;
    final input = List.generate(
      1,
      (_) => List.generate(
        224,
        (y) => List.generate(224, (x) {
          final pixel = resized.getPixel(x, y);
          final channels = [pixel.r, pixel.g, pixel.b];
          return List.generate(
            3,
            (channel) => inputTensor.type == TensorType.float32
                ? (channels[channel] / 255.0 - [0.485, 0.456, 0.406][channel]) /
                    [0.229, 0.224, 0.225][channel]
                : (channels[channel] / 255.0 - [0.485, 0.456, 0.406][channel]) /
                        [0.229, 0.224, 0.225][channel] /
                        inputScale +
                    inputZeroPoint,
          );
        }),
      ),
    );

    final output = List.filled(2, 0.0).reshape([1, 2]);
    _interpreter.run(input, output);
    final outputTensor = _interpreter.getOutputTensor(0);
    final scores = List<double>.from(output[0].map((value) {
      if (outputTensor.type == TensorType.float32) return value.toDouble();
      return (value - outputTensor.params.zeroPoint) * outputTensor.params.scale;
    }));
    final bestIndex = scores[0] >= scores[1] ? 0 : 1;
    return Prediction(label: labels[bestIndex], confidence: scores[bestIndex]);
  }

  void close() => _interpreter.close();
}