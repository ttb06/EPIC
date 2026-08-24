import unittest

from genrec.utils import parse_command_line_args


class CommandLineConfigTest(unittest.TestCase):
    def test_preserves_equals_inside_value(self):
        parsed = parse_command_line_args(['--run_id=ab=c'])
        self.assertEqual(parsed, {'run_id': 'ab=c'})

    def test_converts_common_values(self):
        parsed = parse_command_line_args([
            '--steps=10', '--reproducibility=True', '--topk=[5,10]'])
        self.assertEqual(parsed['steps'], 10)
        self.assertTrue(parsed['reproducibility'])
        self.assertEqual(parsed['topk'], [5, 10])

    def test_rejects_missing_prefix(self):
        with self.assertRaises(ValueError):
            parse_command_line_args(['steps=10'])


if __name__ == '__main__':
    unittest.main()
