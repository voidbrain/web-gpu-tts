// src/app/services/embedder.service.ts
import { Injectable } from '@angular/core';
import { env, pipeline } from '@huggingface/transformers';

interface Command {
  command: string;
  examples: string[];
  params?: string[];
}

interface CommandMatch {
  command: string | null;
  params: Record<string, any>;
}

env.allowLocalModels = true;
env.localModelPath = 'assets/models';
env.backends.onnx!.wasm!.wasmPaths = {
  wasm: '/assets/wasm/ort-wasm-simd-threaded.jsep.wasm',
};

@Injectable({ providedIn: 'root' })
export class EmbedderService {
  private embedder: any | null = null;
  private loadingPromise: Promise<void> | null = null;

  private commands: Command[] = [
    {
      command: 'charge_battery',
      examples: ['Carica batteria {batteryId}', 'Inizia a caricare batteria {batteryId}', 'Charge battery {batteryId}'],
      params: ['batteryId', 'series'],
    },
    {
      command: 'discharge_battery',
      examples: ['Scarica batteria {batteryId}', 'Discharge battery {batteryId}'],
      params: ['batteryId'],
    },
    {
      command: 'check_resistance',
      examples: ['Controlla resistenza batteria {batteryId}', 'Check resistance battery {batteryId}'],
      params: ['batteryId'],
    },
    {
      command: 'store_battery',
      examples: ['Metti in deposito batteria {batteryId}', 'Store battery {batteryId}'],
      params: ['batteryId'],
    },
  ];

  private commandEmbeddings: { command: string; embedding: number[] }[] = [];

  /** Initialize the embedder once */
  async init(): Promise<void> {
    if (this.embedder) return;
    if (this.loadingPromise) return this.loadingPromise;

    this.loadingPromise = (async () => {
      console.log('Loading multilingual MiniLM model...');
      this.embedder = await pipeline('feature-extraction', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', {
        local_files_only: true,
        dtype: 'fp32',
        device: 'webgpu',
      });
      console.log('Embedder ready ✅');

      // Precompute command embeddings
      for (const cmd of this.commands) {
        for (const ex of cmd.examples) {
          const emb = await this.embed(ex);
          this.commandEmbeddings.push({ command: cmd.command, embedding: emb });
        }
      }
      console.log('Command embeddings ready ✅');
    })();

    return this.loadingPromise;
  }

  /** Embed text into a flat number[] */
  async embed(text: string): Promise<number[]> {
    if (!this.embedder) await this.init();
    const result = await this.embedder!(text);

    // Normalize first tensor/array
    return this.normalizeTensor(result[0]);
  }

  /** Convert ORT tensor or nested array to flat number[] */
  private normalizeTensor(tensor: any): number[] {
    if (!tensor) return [];

    // ORT tensor with cpuData
    if (tensor.cpuData) {
      const array = Object.values(tensor.cpuData) as number[];
      const seqLen = tensor.dims[0];
      const hiddenSize = tensor.dims[1];

      return Array.from({ length: hiddenSize }, (_, col) => array.reduce((sum, _, row) => sum + array[row * hiddenSize + col], 0) / seqLen);
    }

    // ORT tensor with flat data array
    if (tensor.data && Array.isArray(tensor.data)) {
      return tensor.data;
    }

    // Nested array [seq_len, hidden_size]
    if (Array.isArray(tensor) && Array.isArray(tensor[0])) {
      const seqLen = tensor.length;
      const hiddenSize = tensor[0].length;

      return Array.from({ length: hiddenSize }, (_, col) => tensor.reduce((sum, row) => sum + row[col], 0) / seqLen);
    }

    // Plain flat array
    if (Array.isArray(tensor)) return tensor;

    // Unknown structure: try flatten recursively
    if (typeof tensor === 'object') {
      const flat: number[] = [];
      const recurse = (obj: any) => {
        if (Array.isArray(obj)) obj.forEach(recurse);
        else if (typeof obj === 'number') flat.push(obj);
        else if (obj && typeof obj === 'object') Object.values(obj).forEach(recurse);
      };
      recurse(tensor);
      if (flat.length) return flat;
    }

    throw new Error('Cannot normalize embedding result: unknown tensor format');
  }

  /** Cosine similarity */
  cosineSim(a: number[], b: number[]): number {
    if (!a.length || !b.length) return 0;
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dot / (magA * magB);
  }

  /** Parse text into a command + parameters */
  async parseCommand(text: string): Promise<CommandMatch> {
    if (!this.commandEmbeddings.length) await this.init();

    const inputEmb = await this.embed(text);

    let bestScore = -1;
    let bestCommand: string | null = null;

    for (const cmd of this.commandEmbeddings) {
      const score = this.cosineSim(inputEmb, cmd.embedding);
      if (score > bestScore) {
        bestScore = score;
        bestCommand = cmd.command;
      }
    }

    const ids = text.match(/\b\d+\b/g)?.map(Number) || [];
    return { command: bestCommand, params: { batteryId: ids.length === 1 ? ids[0] : ids } };
  }

  /** Execute mapped command */
  async executeCommand(result: CommandMatch) {
    if (!result.command) return console.warn('No command matched');

    switch (result.command) {
      case 'charge_battery':
        console.log('Charging battery', result.params['batteryId']);
        break;
      case 'discharge_battery':
        console.log('Discharging battery', result.params['batteryId']);
        break;
      case 'check_resistance':
        console.log('Checking resistance', result.params['batteryId']);
        break;
      case 'store_battery':
        console.log('Storing battery', result.params['batteryId']);
        break;
      default:
        console.warn('Unknown command', result.command);
    }
  }
}
