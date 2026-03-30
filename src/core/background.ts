/**
 * Background task queue.
 *
 * Non-blocking work (embed, store, summarize, compress) gets queued here
 * and runs while the main agent is thinking or the user is reading.
 *
 * Tasks run in order but don't block the critical path.
 * Call flush() to process all pending tasks, or let them run automatically.
 */

import type { BackgroundTask } from '../types.js';

export class BackgroundQueue {
  private queue: BackgroundTask[] = [];
  private processing = false;
  private onError?: (error: unknown, task: BackgroundTask) => void;

  constructor(options?: { onError?: (error: unknown, task: BackgroundTask) => void }) {
    this.onError = options?.onError;
  }

  /** Add a task to the background queue */
  enqueue(task: BackgroundTask): void {
    this.queue.push(task);
  }

  /** Add a task and start processing immediately (fire-and-forget) */
  enqueueAndProcess(task: BackgroundTask): void {
    this.queue.push(task);
    void this.flush();
  }

  /** Process all pending tasks */
  async flush(): Promise<void> {
    if (this.processing) return;
    this.processing = true;

    try {
      while (this.queue.length > 0) {
        const task = this.queue.shift();
        if (!task) break;

        try {
          await task.execute();
        } catch (error) {
          if (this.onError) {
            this.onError(error, task);
          }
          // Background tasks failing shouldn't crash the agent
        }
      }
    } finally {
      this.processing = false;
    }
  }

  /** Check if there are pending tasks */
  get pending(): boolean {
    return this.queue.length > 0;
  }

  /** Number of pending tasks */
  get size(): number {
    return this.queue.length;
  }

  /** Clear all pending tasks */
  clear(): void {
    this.queue = [];
  }
}
