import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:image_picker/image_picker.dart';

import 'classifier.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.classifier});

  final MaizeClassifier classifier;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _picker = ImagePicker();
  Uint8List? _imageBytes;
  Prediction? _prediction;
  bool _busy = false;

  Future<void> _capture(ImageSource source) async {
    final file = await _picker.pickImage(source: source);
    if (file == null) return;
    setState(() => _busy = true);
    final bytes = await file.readAsBytes();
    final prediction = widget.classifier.classify(bytes);
    if (!mounted) return;
    setState(() {
      _imageBytes = bytes;
      _prediction = prediction;
      _busy = false;
    });
    await Hive.box('scan_history').add({
      'label': prediction.label,
      'confidence': prediction.confidence,
      'timestamp': DateTime.now().toIso8601String(),
    });
  }

  @override
  Widget build(BuildContext context) {
    final prediction = _prediction;
    return Scaffold(
      appBar: AppBar(title: const Text('Maize Seed Quality')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          AspectRatio(
            aspectRatio: 1,
            child: DecoratedBox(
              decoration: BoxDecoration(color: Theme.of(context).colorScheme.surfaceContainerHighest),
              child: _imageBytes == null
                  ? const Center(child: Icon(Icons.camera_alt_outlined, size: 64))
                  : Image.memory(_imageBytes!, fit: BoxFit.cover),
            ),
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(child: FilledButton.icon(onPressed: _busy ? null : () => _capture(ImageSource.camera), icon: const Icon(Icons.camera_alt), label: const Text('Camera'))),
              const SizedBox(width: 12),
              Expanded(child: OutlinedButton.icon(onPressed: _busy ? null : () => _capture(ImageSource.gallery), icon: const Icon(Icons.photo_library_outlined), label: const Text('Gallery'))),
            ],
          ),
          if (_busy) const Padding(padding: EdgeInsets.only(top: 24), child: Center(child: CircularProgressIndicator())),
          if (prediction != null) ...[
            const SizedBox(height: 28),
            Text(prediction.label, style: Theme.of(context).textTheme.headlineMedium),
            Text('${(prediction.confidence * 100).toStringAsFixed(1)}% confidence'),
          ],
          const SizedBox(height: 28),
          Text('Recent scans', style: Theme.of(context).textTheme.titleLarge),
          ValueListenableBuilder(
            valueListenable: Hive.box('scan_history').listenable(),
            builder: (context, Box box, _) {
              final entries = box.values.toList().reversed.take(10);
              return Column(
                children: entries.map((entry) {
                  final record = Map<String, dynamic>.from(entry as Map);
                  return ListTile(
                    contentPadding: EdgeInsets.zero,
                    title: Text(record['label'] as String),
                    subtitle: Text(DateTime.parse(record['timestamp'] as String).toLocal().toString()),
                    trailing: Text('${((record['confidence'] as num) * 100).toStringAsFixed(1)}%'),
                  );
                }).toList(),
              );
            },
          ),
        ],
      ),
    );
  }
}