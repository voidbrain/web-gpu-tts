// src/app/services/embedder.service.ts
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

// --- Hugging Face configuration ---
env.allowLocalModels = true;
env.localModelPath = 'assets/models';
env.backends.onnx!.wasm!.wasmPaths = {
  wasm: '/assets/wasm/ort-wasm-simd-threaded.jsep.wasm',
};

@Injectable({ providedIn: 'root' })
export class EmbedderService {
  private embedder: any | null = null;
  private loadingPromise: Promise<void> | null = null;

  // --- Realistic example embeddings (placeholders replaced with sample values) ---
  private commands: Command[] = [
    {
      command: 'charge_battery',
      examples: ['Carica batteria 1', 'Carica batteria 1 serie gialla', 'Charge battery 1', 'Charge battery 1, yellow series'],
      params: ['batteryId', 'series'],
    },
    {
      command: 'discharge_battery',
      examples: ['Scarica batteria 1', 'Discharge battery 1'],
      params: ['batteryId'],
    },
    {
      command: 'check_resistance',
      examples: ['Controlla resistenza batteria 1', 'Check resistance battery 1'],
      params: ['batteryId'],
    },
    {
      command: 'store_battery',
      examples: ['Metti in deposito batteria 1', 'Store battery 1'],
      params: ['batteryId'],
    },
  ];

  private commandEmbeddings: { command: string; embedding: number[] }[] = [];

  /** --- Initialize Hugging Face embedder and precompute embeddings --- */
  async init(): Promise<void> {
    if (this.embedder) return;
    if (this.loadingPromise) return this.loadingPromise;

    this.loadingPromise = (async () => {
      try {
        console.log('[EmbedderService] Loading multilingual MiniLM model...');
        this.embedder = await pipeline('feature-extraction', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', {
          local_files_only: true,
          dtype: 'fp32',
          device: 'webgpu',
        });
        console.log('[EmbedderService] Embedder ready ✅');

        // Precompute command embeddings
        for (const cmd of this.commands) {
          for (const ex of cmd.examples) {
            const emb = await this.embed(ex);
            if (emb.length > 0) {
              this.commandEmbeddings.push({ command: cmd.command, embedding: emb });
              console.log(`[EmbedderService] Embedded example: "${ex}" → len=${emb.length}`);
            } else {
              console.warn(`[EmbedderService] ⚠️ Failed to embed example: "${ex}"`);
            }
          }
        }

        console.log(`[EmbedderService] Command embeddings ready ✅ (${this.commandEmbeddings.length} total)`);
      } catch (err) {
        console.error('[EmbedderService] Initialization error:', err);
        throw err;
      }
    })();

    return this.loadingPromise;
  }

  /** --- Embed text into flat number[] --- */
  async embed(text: string): Promise<number[]> {
    if (!text?.trim()) return [];
    if (!this.embedder) await this.init();

    try {
      const result = await this.embedder!(text);
      const tensor = result?.[0];
      const flat = this.tensorToVector(tensor);

      if (!flat.length) {
        console.warn('[EmbedderService] ⚠️ Empty embedding for text:', text);
        return [];
      }

      const normalized = this.normalizeVector(flat);

      // Debug info
      const mean = normalized.reduce((s, v) => s + v, 0) / normalized.length;
      const variance = normalized.reduce((s, v) => s + (v - mean) ** 2, 0) / normalized.length;
      console.log(`[EmbedderService] Embedded "${text}" → len=${normalized.length}, var=${variance.toFixed(6)}`);

      return normalized;
    } catch (err) {
      console.error('[EmbedderService] Error embedding text:', text, err);
      return [];
    }
  }

  /** --- Convert ORT tensor or nested array to flat number[] --- */
  private tensorToVector(tensor: any): number[] {
    const flat: number[] = [];

    const flatten = (obj: any): void => {
      if (!obj) return;

      // Normal array
      if (Array.isArray(obj)) {
        for (const x of obj) flatten(x);
      }
      // Typed array (Float32Array, Int32Array, etc.)
      else if (ArrayBuffer.isView(obj)) {
        const arr = obj as unknown as number[]; // cast typed array to number[]
        for (let i = 0; i < arr.length; i++) flatten(arr[i]);
      }
      // ORT tensor with cpuData
      else if (typeof obj === 'object') {
        if ('cpuData' in obj && obj.cpuData) {
          const arr = obj.cpuData as Float32Array | number[];
          const seqLen = obj.dims?.[0] ?? 1;
          const hiddenSize = obj.dims?.[1] ?? arr.length;
          for (let col = 0; col < hiddenSize; col++) {
            let sum = 0;
            for (let row = 0; row < seqLen; row++) {
              sum += arr[row * hiddenSize + col];
            }
            flat.push(sum / seqLen);
          }
        } else {
          Object.values(obj).forEach(flatten);
        }
      }
      // Single number
      else if (typeof obj === 'number' && isFinite(obj)) {
        flat.push(obj);
      }
    };

    flatten(tensor);
    return flat;
  }

  /** --- Normalize vector to unit length --- */
  private normalizeVector(vec: number[]): number[] {
    const mag = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
    return mag > 0 ? vec.map(v => v / mag) : vec;
  }

  /** --- Cosine similarity --- */
  cosineSim(a: number[], b: number[]): number {
    if (!a.length || !b.length) return 0;
    let dot = 0,
      normA = 0,
      normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] ** 2;
      normB += b[i] ** 2;
    }
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dot / denominator;
  }

  /** --- Parse command text and extract optional params --- */
  async parseCommand(text: string): Promise<CommandMatch> {
    if (!this.commandEmbeddings.length) await this.init();

    const inputEmb = await this.embed(text);
    if (!inputEmb.length) return { command: null, params: {} };

    let bestScore = -1;
    let bestCommand: string | null = null;

    for (const cmd of this.commandEmbeddings) {
      const score = this.cosineSim(inputEmb, cmd.embedding);
      if (score > bestScore) {
        bestScore = score;
        bestCommand = cmd.command;
      }
    }

    console.log('[EmbedderService] Best match:', bestCommand, 'score:', bestScore.toFixed(3));

    // --- Extract parameters ---
    const ids = text.match(/\b\d+\b/g)?.map(Number) || [];
    const seriesMatch = text.match(/serie\s+(\w+)|(\w+)\s+series/i);
    const series = seriesMatch ? seriesMatch[1] || seriesMatch[2] : undefined;

    return {
      command: bestCommand,
      params: {
        batteryId: ids.length === 1 ? ids[0] : ids,
        series,
      },
    };
  }

  /** --- Execute command --- */
  async executeCommand(result: CommandMatch) {
    if (!result.command) {
      console.warn('[EmbedderService] ⚠️ No command recognized');
      return;
    }

    const id = result.params['batteryId'];
    const series = result.params['series'];
    switch (result.command) {
      case 'charge_battery':
        console.log(`⚡ Charging battery ${id}` + (series ? ` (${series} series)` : ''));
        break;
      case 'discharge_battery':
        console.log(`🔋 Discharging battery ${id}`);
        break;
      case 'check_resistance':
        console.log(`🧪 Checking resistance for battery ${id}`);
        break;
      case 'store_battery':
        console.log(`📦 Storing battery ${id}`);
        break;
      default:
        console.warn('[EmbedderService] Unknown command:', result.command);
    }
  }
}
