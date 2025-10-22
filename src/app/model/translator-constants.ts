export class TranslatorConstants {
  static readonly MODELS: Record<string, number> = {
    // Original checkpoints
    'onnx-community/whisper-small': 586, // 353 + 233
  };

  // List of supported languages:
  // https://help.openai.com/en/articles/7031512-whisper-api-faq
  // https://github.com/openai/whisper/blob/248b6cb124225dd263bb9bd32d060b6517e067f8/whisper/tokenizer.py#L79
  static readonly SUPPORTED_LANGUAGES = {
    en: 'english',
    it: 'italian',
  };
}
