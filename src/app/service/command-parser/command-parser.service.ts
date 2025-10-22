import { Injectable } from '@angular/core';
import * as hf from '@huggingface/transformers';

interface Command {
  command: string;
  examples: string[];
  params?: string[];
}

interface CommandMatch {
  command: string | null;
  params: Record<string, any>;
}

@Injectable({ providedIn: 'root' })
export class CommandService {
  private embedder: any;
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

  /** Initialize the embedding model and precompute command embeddings */
  async init() {
    // Load multilingual embedding model (TFJS)
    this.embedder = await hf.pipeline('feature-extraction', '/assets/models/paraphrase-multilingual-MiniLM-L12-v2');

    // Precompute embeddings for command examples
    for (const cmd of this.commands) {
      for (const ex of cmd.examples) {
        const emb = (await this.embedder(ex))[0];
        this.commandEmbeddings.push({ command: cmd.command, embedding: emb });
      }
    }
  }

  /** Parse typed text into command + parameters */
  async parseCommand(text: string): Promise<CommandMatch> {
    if (!this.embedder) throw new Error('Embedder not initialized');

    const inputEmb = (await this.embedder(text))[0];

    // Cosine similarity search
    let bestScore = -1;
    let bestCommand: string | null = null;

    for (const cmd of this.commandEmbeddings) {
      const score = this.cosineSim(inputEmb, cmd.embedding);
      if (score > bestScore) {
        bestScore = score;
        bestCommand = cmd.command;
      }
    }

    // Extract batteryId(s) from text
    const ids = text.match(/\b\d+\b/g)?.map(Number) || [];

    return { command: bestCommand, params: { batteryId: ids.length === 1 ? ids[0] : ids } };
  }

  /** Execute mapped command (integrate with your battery services) */
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

  /** Cosine similarity helper */
  private cosineSim(a: number[], b: number[]): number {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dot / (magA * magB);
  }
}
