/**
 * The Subconscious — main entry point.
 *
 * Two layers:
 *   SCRIPTED (deterministic): enrich, count tokens, score relevancy, enforce budget
 *   INTELLIGENT (LLM): classify, recall, reshape, compress, summarize
 *
 * Two paths:
 *   CRITICAL (blocking): scripted enrich → classify → maybe recall/reshape → budget → return
 *   BACKGROUND (non-blocking): stash, summarize, recompute relevancy
 */

import type {
  EnrichedMessage,
  IngestResult,
  Message,
  PrepareAction,
  PrepareResult,
  StatusCallback,
  SubconsciousConfig,
} from '../types.js';
import {
  classify,
  compressMessages,
  decideRepresentation,
  recall as recallFromEngine,
  reshape,
  sessionKey,
  stash,
  summarize,
} from './engine.js';
import type { EngineConfig } from './engine.js';
import { BackgroundQueue } from './background.js';
import {
  deduplicate,
  enrich,
  enforceTokenBudget,
  orderChronologically,
  removeExpired,
  totalTokens,
} from './scripted.js';

function generateId(): string {
  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export class Subconscious {
  private readonly config: EngineConfig;
  private readonly onStatus: StatusCallback;
  private readonly background: BackgroundQueue;
  private context: EnrichedMessage[] = [];
  private summary = '';
  private turn = 0;

  constructor(config: SubconsciousConfig) {
    this.config = {
      vector: config.vector,
      kv: config.kv,
      llm: config.llm,
      sessionId: config.sessionId ?? generateId(),
      tokenBudget: config.tokenBudget ?? 8000,
      recentWindow: config.recentWindow ?? 10,
    };

    this.onStatus = config.onStatus ?? (() => {});

    this.background = new BackgroundQueue({
      onError: (error) => {
        console.error('[Subconscious] Background task failed:', error);
      },
    });
  }

  // =================================================================
  // CRITICAL PATH
  // =================================================================

  /**
   * Prepare — process a new inbound message and return curated context.
   *
   * HARD RULE: the current user message is NEVER modified.
   */
  async prepare(rawMessage: Message): Promise<PrepareResult> {
    const actions: PrepareAction[] = [];

    // --- SCRIPTED: enrich the message (tokens, priority, source) ---
    const newMessage = enrich(rawMessage, this.turn);
    this.turn++;

    // --- SCRIPTED: remove expired messages from context ---
    this.context = removeExpired(this.context);

    // --- INTELLIGENT: classify — what does this message need? ---
    this.onStatus({ phase: 'classifying', message: 'Analyzing message...' });

    const classification = await classify(
      this.config,
      newMessage,
      this.summary,
      totalTokens(this.context),
    );

    let recalledCount = 0;
    let compressedCount = 0;

    switch (classification.action) {
      case 'passthrough': {
        this.context.push(newMessage);
        actions.push({ type: 'passthrough', detail: classification.reasoning });
        break;
      }

      case 'recall': {
        this.onStatus({ phase: 'recalling', message: 'Recalling context...' });
        const recalled = await recallFromEngine(this.config, newMessage);
        recalledCount = recalled.length;

        // --- SCRIPTED: deduplicate (don't inject what's already in context) ---
        this.context.push(...recalled, newMessage);
        this.context = deduplicate(this.context);

        actions.push({
          type: 'recall',
          detail: `Recalled ${recalled.length} messages: ${classification.reasoning}`,
        });
        break;
      }

      case 'reshape': {
        this.onStatus({
          phase: 'reshaping',
          message: 'Conversation shifted — reshaping context...',
          progress: 0,
        });

        const reshapeResult = await reshape(
          this.config,
          newMessage,
          this.context,
          this.summary,
          this.turn,
        );

        this.onStatus({ phase: 'reshaping', message: 'Context reshaped.', progress: 1 });

        compressedCount = this.context.length;
        this.context = [...reshapeResult.messages, newMessage];

        actions.push({ type: 'reshape', detail: classification.reasoning });
        actions.push({ type: 'briefing', detail: reshapeResult.briefing.content });
        break;
      }
    }

    // --- SCRIPTED: enforce token budget ---
    const { kept, dropped } = enforceTokenBudget(
      this.context,
      this.config.tokenBudget,
      this.config.recentWindow,
    );
    if (dropped.length > 0) {
      this.context = kept;
      actions.push({
        type: 'budget-trim',
        detail: `Dropped ${dropped.length} messages (${totalTokens(dropped)} tokens) to stay within budget`,
      });
    }

    // --- SCRIPTED: order chronologically ---
    this.context = orderChronologically(this.context);

    this.onStatus({ phase: 'ready', message: 'Ready.' });

    // --- BACKGROUND: queue embed, store, summarize ---
    this.background.enqueueAndProcess({
      type: 'embed',
      execute: () => stash(this.config, newMessage),
    });

    this.background.enqueue({
      type: 'summarize',
      execute: async () => {
        this.summary = await summarize(this.config, this.summary, [newMessage]);
      },
    });

    // Background: compress if over budget after recent window is excluded
    const nonRecent = this.context.length - this.config.recentWindow;
    if (nonRecent > 20) {
      this.background.enqueue({
        type: 'compress',
        execute: () => this.compressOldMessages(),
      });
    }

    return {
      messages: [...this.context],
      actions,
      classification: classification.action,
      summary: this.summary,
      recalled: recalledCount,
      compressed: compressedCount,
      totalTokens: totalTokens(this.context),
      pendingBackground: this.background.pending,
    };
  }

  // =================================================================
  // INGEST
  // =================================================================

  /**
   * Ingest — process the main agent's response.
   * Stores full version for recall. Decides context representation.
   */
  async ingest(rawResponse: Message): Promise<IngestResult> {
    const message = enrich(
      { ...rawResponse, id: rawResponse.id || generateId(), timestamp: rawResponse.timestamp || Date.now() },
      this.turn,
    );
    this.turn++;

    // Decide: keep full in context or summarize?
    const { message: contextMessage, summarized } = await decideRepresentation(
      this.config,
      message,
    );

    this.context.push(contextMessage);

    // Background: store FULL version (not summarized) for recall
    this.background.enqueueAndProcess({
      type: 'store',
      execute: () => stash(this.config, message),
    });

    return {
      representation: summarized ? 'summarized' : 'full',
      pendingBackground: this.background.pending,
    };
  }

  // =================================================================
  // RECALL — explicit tool for the main agent
  // =================================================================

  async recall(query: string, topK: number = 5): Promise<EnrichedMessage[]> {
    const queryMessage = enrich(
      { id: 'recall-query', role: 'user', content: query, timestamp: Date.now() },
      this.turn,
    );
    return recallFromEngine(this.config, queryMessage, topK);
  }

  // =================================================================
  // BACKGROUND MANAGEMENT
  // =================================================================

  async flush(): Promise<void> {
    await this.background.flush();
  }

  // =================================================================
  // ACCESSORS
  // =================================================================

  get sessionId(): string {
    return this.config.sessionId;
  }

  get currentTurn(): number {
    return this.turn;
  }

  async getSummary(): Promise<string> {
    if (this.summary) return this.summary;
    return (await this.config.kv.get<string>(sessionKey(this.config.sessionId, 'summary'))) ?? '';
  }

  getContext(): EnrichedMessage[] {
    return [...this.context];
  }

  get hasPendingWork(): boolean {
    return this.background.pending;
  }

  // =================================================================
  // INTERNAL
  // =================================================================

  private async compressOldMessages(): Promise<void> {
    const recentStart = Math.max(0, this.context.length - this.config.recentWindow);
    const older = this.context.slice(0, recentStart);
    const recent = this.context.slice(recentStart);

    // Don't compress pinned messages
    const toCompress = older.filter((m) => !m.meta.pinned);
    const pinned = older.filter((m) => m.meta.pinned);

    if (toCompress.length < 5) return; // not worth compressing

    const compressed = await compressMessages(this.config, toCompress, undefined, this.turn);
    this.context = [...pinned, compressed, ...recent];
  }
}
