"""
Test Suite for CV Pipeline
Run with: python test_installation.py
Or use pytest: pytest test_installation.py -v
"""

import sys
import unittest
import numpy as np


class TestDependencies(unittest.TestCase):
    """Test suite for checking package dependencies"""
    
    def test_numpy_import(self):
        """Test numpy can be imported"""
        try:
            import numpy
            self.assertTrue(True)
        except ImportError:
            self.fail("numpy not installed")
    
    def test_scipy_import(self):
        """Test scipy can be imported"""
        try:
            import scipy
            self.assertTrue(True)
        except ImportError:
            self.fail("scipy not installed")
    
    def test_skimage_import(self):
        """Test scikit-image can be imported"""
        try:
            import skimage
            self.assertTrue(True)
        except ImportError:
            self.fail("scikit-image not installed")
    
    def test_opencv_import(self):
        """Test opencv-python can be imported"""
        try:
            import cv2
            self.assertTrue(True)
        except ImportError:
            self.fail("opencv-python not installed")
    
    def test_matplotlib_import(self):
        """Test matplotlib can be imported"""
        try:
            import matplotlib
            self.assertTrue(True)
        except ImportError:
            self.fail("matplotlib not installed")
    
    def test_pandas_import(self):
        """Test pandas can be imported"""
        try:
            import pandas
            self.assertTrue(True)
        except ImportError:
            self.fail("pandas not installed")


class TestPipelineImport(unittest.TestCase):
    """Test suite for pipeline module imports"""
    
    def test_hybrid_pipeline_import(self):
        """Test hybrid_pipeline module can be imported"""
        try:
            import hybrid_pipeline
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Cannot import hybrid_pipeline: {e}")
    
    def test_segmentation_pipeline_import(self):
        """Test HybridSegmentationPipeline class can be imported"""
        try:
            from hybrid_pipeline import HybridSegmentationPipeline
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Cannot import HybridSegmentationPipeline: {e}")
    
    def test_growth_analyzer_import(self):
        """Test GrowthAnalyzer class can be imported"""
        try:
            from hybrid_pipeline import GrowthAnalyzer
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Cannot import GrowthAnalyzer: {e}")
    
    def test_config_import(self):
        """Test config module can be imported"""
        try:
            import config
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Cannot import config: {e}")


class TestPipelineInstantiation(unittest.TestCase):
    """Test suite for pipeline class instantiation"""
    
    def test_segmentation_pipeline_instantiation(self):
        """Test HybridSegmentationPipeline can be instantiated"""
        from hybrid_pipeline import HybridSegmentationPipeline
        pipeline = HybridSegmentationPipeline()
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.gaussian_sigma, 1.0)

    def test_segmentation_pipeline_custom_params(self):
        """Test HybridSegmentationPipeline with custom parameters"""
        from hybrid_pipeline import HybridSegmentationPipeline
        pipeline = HybridSegmentationPipeline(
            gaussian_sigma=2.0,
            sobel_ksize=5,
        )
        self.assertEqual(pipeline.gaussian_sigma, 2.0)
        self.assertEqual(pipeline.sobel_ksize, 5)
    
    def test_growth_analyzer_instantiation(self):
        """Test GrowthAnalyzer can be instantiated"""
        from hybrid_pipeline import GrowthAnalyzer
        analyzer = GrowthAnalyzer()
        self.assertIsNotNone(analyzer)


class TestPreprocessing(unittest.TestCase):
    """Test suite for preprocessing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hybrid_pipeline import HybridSegmentationPipeline
        self.pipeline = HybridSegmentationPipeline()
        self.test_frame = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    def test_preprocess_frame_returns_tuple(self):
        """Test preprocess_frame returns blurred and edges"""
        blurred, edges = self.pipeline.preprocess_frame(self.test_frame)
        self.assertIsNotNone(blurred)
        self.assertIsNotNone(edges)
    
    def test_preprocess_frame_shape(self):
        """Test preprocessed output has correct shape"""
        blurred, edges = self.pipeline.preprocess_frame(self.test_frame)
        self.assertEqual(edges.shape, self.test_frame.shape)
    
    def test_preprocess_gaussian_effect(self):
        """Test Gaussian blur reduces variance"""
        blurred, _ = self.pipeline.preprocess_frame(self.test_frame)
        # Blurred image should have lower variance
        self.assertLessEqual(np.var(blurred), np.var(self.test_frame))


class TestMemoryMask(unittest.TestCase):
    """Test suite for memory mask functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hybrid_pipeline import HybridSegmentationPipeline
        self.pipeline = HybridSegmentationPipeline()
        self.test_mask = np.zeros((512, 512), dtype=np.uint16)
        self.test_mask[200:300, 200:300] = 1
    
    def test_memory_mask_reset(self):
        """Test memory mask reset functionality"""
        memory = self.pipeline.update_memory_mask(self.test_mask, reset=True)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.shape, self.test_mask.shape)
    
    def test_memory_mask_accumulation(self):
        """Test memory mask accumulates regions"""
        # First frame
        mask1 = np.zeros((512, 512), dtype=np.uint16)
        mask1[100:200, 100:200] = 1
        memory1 = self.pipeline.update_memory_mask(mask1, reset=True)
        
        # Second frame with different region
        mask2 = np.zeros((512, 512), dtype=np.uint16)
        mask2[300:400, 300:400] = 1
        memory2 = self.pipeline.update_memory_mask(mask2, reset=False)
        
        # Memory should have both regions
        self.assertGreater(np.sum(memory2), np.sum(mask1) + np.sum(mask2) - 10000)


class TestGrowthAnalysis(unittest.TestCase):
    """Test suite for growth analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hybrid_pipeline import GrowthAnalyzer
        self.analyzer = GrowthAnalyzer(rolling_window=3, interval_minutes=2.0)
        self.test_masks = [
            np.zeros((512, 512), dtype=np.uint16),
            np.zeros((512, 512), dtype=np.uint16),
            np.zeros((512, 512), dtype=np.uint16)
        ]
        self.test_masks[0][200:250, 200:250] = 1
        self.test_masks[1][200:260, 200:260] = 1
        self.test_masks[2][200:270, 200:270] = 1
    
    def test_compute_area_growth_length(self):
        """Test area growth returns correct length"""
        areas = self.analyzer.compute_area_growth(self.test_masks)
        self.assertEqual(len(areas), len(self.test_masks))
    
    def test_compute_area_growth_increasing(self):
        """Test area growth is increasing"""
        areas = self.analyzer.compute_area_growth(self.test_masks)
        self.assertLessEqual(areas[0], areas[1])
        self.assertLessEqual(areas[1], areas[2])
    
    def test_compute_growth_rate_output(self):
        """Test growth rate computation"""
        areas = self.analyzer.compute_area_growth(self.test_masks)
        growth_rates = self.analyzer.compute_growth_rate_rolling(areas)
        self.assertIsInstance(growth_rates, np.ndarray)


class TestConfig(unittest.TestCase):
    """Test suite for configuration"""
    
    def test_config_has_paths(self):
        """Test config has required path variables"""
        import config
        self.assertTrue(hasattr(config, 'BASE_DIR'))
        self.assertTrue(hasattr(config, 'REF_RAW_DIR'))
        self.assertTrue(hasattr(config, 'OUTPUT_DIR'))
    
    def test_config_has_parameters(self):
        """Test config has required parameter variables"""
        import config
        self.assertTrue(hasattr(config, 'INTERVAL_MINUTES'))
        self.assertTrue(hasattr(config, 'PIXEL_SIZE_UM'))
        self.assertTrue(hasattr(config, 'ROLLING_WINDOW'))
    
    def test_config_parameter_types(self):
        """Test config parameters have correct types"""
        import config
        self.assertIsInstance(config.INTERVAL_MINUTES, float)
        self.assertIsInstance(config.PIXEL_SIZE_UM, float)
        self.assertIsInstance(config.ROLLING_WINDOW, int)


def print_system_info():
    """Print system information"""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except:
        pass
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except:
        pass
    
    try:
        import skimage
        print(f"scikit-image version: {skimage.__version__}")
    except:
        pass
    
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HYBRID CV PIPELINE - TEST SUITE")
    print("="*70)
    print("\nRunning comprehensive test suite...")
    print("Tests organized into categories:")
    print("  • Dependencies (6 tests)")
    print("  • Pipeline Import (4 tests)")
    print("  • Instantiation (3 tests)")
    print("  • Preprocessing (3 tests)")
    print("  • Memory Mask (2 tests)")
    print("  • Growth Analysis (3 tests)")
    print("  • Configuration (3 tests)")
    print("\n" + "="*70)
    
    # Print system info first
    print_system_info()
    
    # Run unittest suite
    print("\n" + "="*70)
    print("RUNNING TESTS")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDependencies))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineImport))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineInstantiation))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryMask))
    suite.addTests(loader.loadTestsFromTestCase(TestGrowthAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:  {len(result.failures)}")
    print(f"Errors:    {len(result.errors)}")
    print("="*70)
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou're ready to run the pipeline!")
        print("\nNext steps:")
        print("  1. Update paths in config.py")
        print("  2. Run example_usage.py to test on real data")
        print("  3. Run parameter_tuner.py to optimize parameters")
        print("  4. Run analysis_notebook.py for full analysis")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above before running the pipeline.")
        if result.failures:
            print("\nFailed tests:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    print("\n" + "="*70)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
