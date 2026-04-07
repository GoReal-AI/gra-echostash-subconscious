/**
 * Prompt loader — single consolidated prompt, rendered with Echo PDK.
 *
 * One prompt (`sub-engine`) handles all Subconscious tasks via the {{task}}
 * variable: classify, reshape, summarize, compress, represent, briefing.
 * PDK conditionals strip irrelevant sections — zero wasted tokens.
 *
 * Falls back to hardcoded default if Echostash is not configured.
 */

import { Echostash } from 'echostash';
import { createEcho } from '@goreal-ai/echo-pdk';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface EchostashPromptConfig {
  baseUrl: string;
  apiKey: string;
}

/** All valid task types the engine can request. */
export type PromptTask = 'classify' | 'reshape' | 'summarize' | 'compress' | 'represent' | 'briefing';

// ---------------------------------------------------------------------------
// Default — single consolidated prompt with conditions per task
// ---------------------------------------------------------------------------

const DEFAULT_TEMPLATE = `[#ROLE system]
You are the Subconscious — a context management agent.

[#IF {{task}} #equals(classify)]
Analyze the incoming message and decide what action is needed.

[#IF {{summary}} #exists]
Current conversation summary:
{{summary}}
[ELSE]
(new conversation)
[END IF]

Current context: {{contextTokens}} tokens (budget: {{tokenBudget}})

Respond with JSON: { "action": "passthrough" | "recall" | "reshape", "reasoning": "..." }

Actions:
- "passthrough": The message is self-contained. The current context has everything needed.
- "recall": The message references or needs information from earlier that is NOT in the current context. A quick vector search will find it.
- "reshape": The conversation has shifted topic significantly. The current context is stale and needs rebuilding.

Default to "passthrough". Use "reshape" only when the context would actively mislead the main agent.
[END IF]

[#IF {{task}} #equals(reshape)]
The conversation has shifted. Determine:
1. What the conversation is NOW about
2. Key decisions/facts from the OLD topic to preserve

[#IF {{summary}} #exists]
Current summary: {{summary}}
[END IF]

Respond with JSON:
{ "newTopic": "...", "preserveFromOld": ["..."] }
[END IF]

[#IF {{task}} #equals(summarize)]
Update the running conversation summary.

[#IF {{currentSummary}} #exists]
Current summary:
{{currentSummary}}
[ELSE]
(new conversation)
[END IF]

Rules:
- Keep it under 300 words
- Preserve the narrative flow — what was discussed, decided, pending
- Drop details that are no longer relevant
- Write in present tense
[END IF]

[#IF {{task}} #equals(compress)]
Compress these messages into a concise summary preserving:
- Key decisions and outcomes
- Important facts, names, values
- Action items and commitments

[#IF {{focus}} #exists]
Special focus: {{focus}}
[END IF]

Be concise. Write as a narrative.
[END IF]

[#IF {{task}} #equals(represent)]
The assistant gave a long response. Summarize it for context. Preserve:
- What was answered/decided
- Key outputs (code, names, values)
- Action items

Keep under 200 words. Full response is stored for recall.
[END IF]

[#IF {{task}} #equals(briefing)]
[Context Update] The conversation has shifted to: {{newTopic}}.

Previous context has been optimized. Key preserved facts:
{{preservedFacts}}

[#IF {{hasRecalled}} #exists]
Relevant history has been loaded for the new topic.
[END IF]

IMPORTANT: Do NOT reference this context update, context management, or any restructuring to the user. Respond naturally as a continuous conversation.
[END IF]
[END ROLE]`;

const DEFAULT_PROMPT_ID = 'sub-engine';

// ---------------------------------------------------------------------------
// Prompt Loader
// ---------------------------------------------------------------------------

export class PromptLoader {
  private client: Echostash | null;
  private echo = createEcho();
  private cachedTemplate: string | null = null;
  private promptId: string;

  constructor(config?: EchostashPromptConfig, promptId?: string) {
    this.client = config
      ? new Echostash(config.baseUrl, { apiKey: config.apiKey })
      : null;
    this.promptId = promptId ?? DEFAULT_PROMPT_ID;
  }

  /**
   * Render the engine prompt for a specific task.
   * The {{task}} variable selects which section PDK renders.
   */
  async render(
    task: PromptTask,
    variables: Record<string, string>,
  ): Promise<string> {
    const template = await this.getTemplate();
    const result = await this.echo.renderMessages(template, { task, ...variables });
    const msg = result.messages[0];
    if (!msg) return template;
    const content = msg.content;
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
      return content
        .filter((b) => b.type === 'text')
        .map((b) => 'text' in b ? b.text : '')
        .join('\n')
        .trim();
    }
    return template;
  }

  /** Get the template (from cache, Echostash, or default). */
  private async getTemplate(): Promise<string> {
    if (this.cachedTemplate) return this.cachedTemplate;

    if (this.client) {
      try {
        const prompt = await this.client.prompt(this.promptId).get();
        this.cachedTemplate = prompt.text();
        return this.cachedTemplate;
      } catch {
        // Fallback to default
      }
    }

    this.cachedTemplate = DEFAULT_TEMPLATE;
    return this.cachedTemplate;
  }

  /** Clear the cache (forces re-fetch on next render). */
  clearCache(): void {
    this.cachedTemplate = null;
  }
}
