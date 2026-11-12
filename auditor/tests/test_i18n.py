import tempfile
import unittest
from pathlib import Path
import i18n


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.old_config = i18n._CONFIG
        self.old_language = i18n._LANGUAGE_CACHE
        self.temp_dir = tempfile.TemporaryDirectory()
        i18n._CONFIG = Path(self.temp_dir.name) / 'settings.json'
        i18n._LANGUAGE_CACHE = None

    def tearDown(self):
        i18n._CONFIG = self.old_config
        i18n._LANGUAGE_CACHE = self.old_language
        self.temp_dir.cleanup()

    def test_english_is_default_and_translates_interface(self):
        self.assertEqual(i18n.get_language(), 'en')
        self.assertEqual(i18n.tr('1. Diseño'), '1. Design')
        self.assertEqual(i18n.tr('Archivo'), 'File')

    def test_spanish_can_be_selected_and_persisted(self):
        i18n.set_language('es')
        self.assertEqual(i18n.get_language(), 'es')
        self.assertEqual(i18n.tr('1. Diseño'), '1. Diseño')
        self.assertEqual(i18n.tr('Language'), 'Idioma')
        i18n._LANGUAGE_CACHE = None
        self.assertEqual(i18n.get_language(), 'es')


if __name__ == '__main__':
    unittest.main()
