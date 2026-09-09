import 'package:flutter/material.dart';
import 'package:hive_flutter/hive_flutter.dart';

import 'classifier.dart';
import 'home_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Hive.initFlutter();
  await Hive.openBox('scan_history');
  final classifier = await MaizeClassifier.load();
  runApp(MaizeApp(classifier: classifier));
}

class MaizeApp extends StatelessWidget {
  const MaizeApp({super.key, required this.classifier});

  final MaizeClassifier classifier;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Maize Seed Quality',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xff287a55)), useMaterial3: true),
      home: HomeScreen(classifier: classifier),
    );
  }
}