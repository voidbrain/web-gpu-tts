import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class TranscriberConfigStorage {
  samplingRate = 16000;
  model = 'onnx-community/whisper-small';
  subtask = 'transcribe';
  language = 'english';
  isMultilingual = true;

  constructor() {
    this.model = 'onnx-community/whisper-small';
  }
}
