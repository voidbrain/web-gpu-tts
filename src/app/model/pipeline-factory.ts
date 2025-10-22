import { env, pipeline } from '@huggingface/transformers';

env.allowLocalModels = true; // ✅ allow local files
env.localModelPath = 'assets/models'; // ✅ base path for models

env.backends.onnx!.wasm!.wasmPaths = {
  wasm: 'assets/wasm/ort-wasm-simd-threaded.jsep.wasm',
};

// Define model factories
// Ensures only one model is created of each type
export class PipelineFactory {
  static task: any = 'automatic-speech-recognition';
  static model: string | null = null;
  static instance: any = null;

  static async getInstance(progressCallback?: (data: any) => void) {
    const localModelPath = 'whisper-small';
    if (this.instance === null && localModelPath) {
      this.instance = pipeline(this.task, localModelPath, {
        dtype: {
          encoder_model: 'fp32',
          decoder_model_merged: 'q4', // or 'fp32' ('fp16' is broken)
        },
        device: 'webgpu',
        progress_callback: progressCallback,
        local_files_only: true,
      });
    }

    return this.instance;
  }
}
