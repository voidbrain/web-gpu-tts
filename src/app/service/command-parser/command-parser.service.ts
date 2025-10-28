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

  // --- Command definitions ---
  private commands: Command[] = [
    {
      command: 'charge_battery',
      examples: [
        'Carica la batteria {batteryId} {series}',
        'Carica la batteria {series} {batteryId}',
        'Carica la batteria {series} numero {batteryId}',
        'Carica la batteria numero {batteryId} {series}',
        'Carica la batteria {batteryId} serie {series}',
        'Charge battery {batteryId} {series}',
        'Charge battery {series} {batteryId}',
        'Charge battery {series} number {batteryId}',
        'Charge battery {batteryId} {series} series',
        'Charge battery {batteryId} series {series}',
        'Charge {series} battery {batteryId}',
      ],
      params: ['batteryId', 'series'],
    },
    {
      command: 'discharge_battery',
      examples: [
        'Scarica la batteria {batteryId} {series}',
        'Scarica la batteria {series} numero {batteryId}',
        'Scarica la batteria {batteryId} serie {series}',
        'Discharge battery {batteryId} {series}',
        'Discharge battery {series} number {batteryId}',
        'Discharge battery {batteryId} {series} series',
        'Discharge battery {batteryId} series {series}',
      ],
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

      return this.normalizeVector(flat);
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
        const arr = obj as Float32Array | Int32Array | Uint8Array | number[];
        for (const val of arr) {
          if (typeof val === 'number' && isFinite(val)) flat.push(val);
          else flatten(val);
        }
      }
      // ORT tensor with cpuData
      else if (typeof obj === 'object') {
        if ('cpuData' in obj && obj.cpuData) {
          const arr = obj.cpuData as Float32Array | number[];
          // for (let i = 0; i < arr.length; i++) {
          for (const val of arr) {
            flat.push(val);
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

  /** --- Parse command text with robust series extraction --- */
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

    // --- Extract batteryId ---
    const ids = text.match(/\b\d+\b/g)?.map(Number) || [];
    const batteryId = ids.length === 1 ? ids[0] : ids;

    // --- Extract series/color robustly ---
    const seriesList = ['rossa', 'blu', 'gialla', 'yellow', 'red', 'green', 'blue'];
    let series: string | undefined;

    const words = text.toLowerCase().split(/\s+/);
    for (const w of words) {
      if (seriesList.includes(w)) {
        series = w;
        break;
      }
    }

    // Fallback: word immediately after battery number
    if (!series && batteryId) {
      const match = text.match(new RegExp(`\\b${batteryId}\\b\\s+(\\w+)`, 'i'));
      if (match) series = match[1].toLowerCase();
    }

    return { command: bestCommand, params: { batteryId, series } };
  }

  /** --- Execute command --- */
  async executeCommand(result: CommandMatch) {
    if (!result.command) {
      console.warn('[EmbedderService] ⚠️ No command recognized');
      return;
    }

    const { batteryId, series } = result.params;
    switch (result.command) {
      case 'charge_battery':
        console.log(`⚡ Charging battery ${batteryId}` + (series ? ` (${series} series)` : ''));
        break;
      case 'discharge_battery':
        console.log(`🔋 Discharging battery ${batteryId}`);
        break;
      case 'check_resistance':
        console.log(`🧪 Checking resistance for battery ${batteryId}`);
        break;
      case 'store_battery':
        console.log(`📦 Storing battery ${batteryId}`);
        break;
      default:
        console.warn('[EmbedderService] Unknown command:', result.command);
    }
  }
}
