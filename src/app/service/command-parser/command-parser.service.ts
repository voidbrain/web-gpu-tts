// src/app/services/command-parser.service.ts
import { Injectable } from '@angular/core';
import { env, pipeline } from '@huggingface/transformers';

interface Command {
  command: string;
  examples: string[];
  params?: string[];
}

export interface CommandMatch {
  command: string | null;
  params: Record<string, any>;
}

// Configure Hugging Face
env.allowLocalModels = true;
env.localModelPath = 'assets/models';
env.backends.onnx!.wasm!.wasmPaths = {
  wasm: '/assets/wasm/ort-wasm-simd-threaded.jsep.wasm',
};

@Injectable({ providedIn: 'root' })
export class CommandService {
  private commands: Command[] = [
    {
      command: 'charge_battery',
      examples: [
        'Carica batteria {batteryId}',
        'Carica batteria {batteryId} serie {serieText}',
        'Inizia a caricare batteria {batteryId}',
        'Charge battery {batteryId}',
        'Start charging battery {batteryId}',
        'Battery {batteryId} charging',
      ],
      params: ['batteryId', 'serieText'],
    },
    {
      command: 'discharge_battery',
      examples: [
        'Scarica batteria {batteryId}',
        'Discharge battery {batteryId}',
        'Start discharging battery {batteryId}',
        'Battery {batteryId} discharging',
      ],
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
  private embedder: any | null = null;
  private loadingPromise: Promise<void> | null = null;

  /** Initialize embedder and precompute embeddings */
  async init(): Promise<void> {
    if (this.embedder) return;
    if (this.loadingPromise) return this.loadingPromise;

    this.loadingPromise = (async () => {
      try {
        console.log('Loading multilingual MiniLM model...');
        this.embedder = await pipeline('feature-extraction', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', {
          local_files_only: true,
          dtype: 'fp32',
          device: 'webgpu', // safer for testing
        });
        console.log('Embedder ready ✅');

        // Precompute embeddings for command examples
        for (const cmd of this.commands) {
          for (const ex of cmd.examples) {
            const emb = await this.getEmbedding(ex);
            this.commandEmbeddings.push({ command: cmd.command, embedding: this.normalizeVec(emb) });
          }
        }

        console.log('Command embeddings precomputed ✅');
      } catch (err) {
        console.error('Error initializing embedder:', err);
      }
    })();

    return this.loadingPromise;
  }

  /** Get embedding for a text */
  private async getEmbedding(text: string): Promise<number[]> {
    if (!this.embedder) throw new Error('Embedder not initialized');

    try {
      const result = await this.embedder(text);
      const tensor = result[0];

      if (tensor.cpuData) {
        const array = Object.values(tensor.cpuData) as number[];
        const seqLen = tensor.dims[0];
        const hiddenSize = tensor.dims[1];

        // Average over tokens to get sentence embedding
        return Array.from({ length: hiddenSize }, (_, col) => array.reduce((sum, _, row) => sum + array[row * hiddenSize + col], 0) / seqLen);
      }

      // fallback if plain array
      if (Array.isArray(tensor)) return tensor as number[];
      return Array(tensor.dims?.[1] || 384).fill(0); // fallback to zero vector
    } catch (err) {
      console.error('Error computing embedding for text:', text, err);
      return Array(384).fill(0); // fallback zero vector
    }
  }

  /** Normalize vector for cosine similarity */
  private normalizeVec(vec: number[]): number[] {
    if (!Array.isArray(vec) || vec.length === 0) return [];
    const mag = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
    return vec.map(v => v / (mag || 1e-10));
  }

  /** Cosine similarity */
  private cosineSim(a: number[], b: number[]): number {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) return -1;
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dot / (magA * magB || 1e-10);
  }

  /** Parse input text to command + params */
  async parseCommand(text: string, threshold = 0.7): Promise<CommandMatch> {
    if (!this.embedder) await this.init();

    const inputVec = this.normalizeVec(await this.getEmbedding(text));
    let bestScore = -1;
    let bestCommand: string | null = null;

    for (const cmd of this.commandEmbeddings) {
      const score = this.cosineSim(inputVec, cmd.embedding);
      if (score > bestScore) {
        bestScore = score;
        bestCommand = cmd.command;
      }
    }

    if (bestScore < threshold) bestCommand = null;

    // Extract batteryId(s)
    const batteryIds = text.match(/\b\d+\b/g)?.map(Number) || [];

    // Extract optional serieText
    const serieMatch = text.match(/serie (\w+)/i);
    const serieText = serieMatch ? serieMatch[1] : undefined;

    console.log('Best command match:', bestCommand, 'with score', bestScore);

    return {
      command: bestCommand,
      params: {
        batteryId: batteryIds.length === 1 ? batteryIds[0] : batteryIds,
        serieText,
      },
    };
  }

  /** Execute command */
  async executeCommand(result: CommandMatch) {
    if (!result.command) return console.warn('No command matched');

    switch (result.command) {
      case 'charge_battery':
        console.log('Charging battery', result.params['batteryId'], result.params['serieText'] || '');
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
