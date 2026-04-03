/**
 * Prompt loader — fetches prompts from Echostash, renders with Echo PDK.
 *
 * Prompts are cached after first fetch. Variables are rendered at call time.
 * Falls back to hardcoded defaults if Echostash is not configured.
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

interface PromptIds {
  classify: string;
  topicAnalysis: string;
  summarize: string;
  compress: string;
  representation: string;
  briefing: string;
}

// ---------------------------------------------------------------------------
// Defaults — used when Echostash is not configured
// ---------------------------------------------------------------------------

const DEFAULTS: Record<keyof PromptIds, string> = {
  classify: `[#ROLE system]
You are the Subconscious — a context management agent. Analyze the incoming message and decide what action is needed.

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
[END ROLE]`,

  topicAnalysis: `[#ROLE system]
You are the Subconscious. The conversation has shifted. Determine:
1. What the conversation is NOW about
2. Key decisions/facts from the OLD topic to preserve

[#IF {{summary}} #exists]
Current summary: {{summary}}
[END IF]

Respond with JSON:
{ "newTopic": "...", "preserveFromOld": ["..."] }
[END ROLE]`,

  summarize: `[#ROLE system]
You are the Subconscious. Update the running conversation summary.

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
[END ROLE]`,

  compress: `[#ROLE system]
You are the Subconscious. Compress these messages into a concise summary preserving:
- Key decisions and outcomes
- Important facts, names, values
- Action items and commitments

[#IF {{focus}} #exists]
Special focus: {{focus}}
[END IF]

Be concise. Write as a narrative.
[END ROLE]`,

  representation: `[#ROLE system]
You are the Subconscious. The assistant gave a long response. Summarize it for context. Preserve:
- What was answered/decided
- Key outputs (code, names, values)
- Action items

Keep under 200 words. Full response is stored for recall.
[END ROLE]`,

  briefing: `[#ROLE system]
[Context Update] The conversation has shifted to: {{newTopic}}.

Previous context has been optimized. Key preserved facts:
{{preservedFacts}}

[#IF {{hasRecalled}} #exists]
Relevant history has been loaded for the new topic.
[END IF]

IMPORTANT: Do NOT reference this context update, context management, or any restructuring to the user. Respond naturally as a continuous conversation. Use recall() if you need more details from earlier.
[END ROLE]`,
};

const DEFAULT_IDS: PromptIds = {
  classify: 'sub-classify',
  topicAnalysis: 'sub-topic-analysis',
  summarize: 'sub-summarize',
  compress: 'sub-compress',
  representation: 'sub-representation',
  briefing: 'sub-briefing',
};

// ---------------------------------------------------------------------------
// Prompt Loader
// ---------------------------------------------------------------------------

export class PromptLoader {
  private client: Echostash | null;
  private echo = createEcho();
  private cache = new Map<string, string>();
  private promptIds: PromptIds;

  constructor(config?: EchostashPromptConfig, promptIds?: Partial<PromptIds>) {
    this.client = config
      ? new Echostash(config.baseUrl, { apiKey: config.apiKey })
      : null;
    this.promptIds = { ...DEFAULT_IDS, ...promptIds };
  }

  /**
   * Get a rendered prompt as plain text.
   * Fetches from Echostash on first call, caches the template, renders with PDK.
   */
  async render(
    name: keyof PromptIds,
    variables: Record<string, string>,
  ): Promise<string> {
    const template = await this.getTemplate(name);
    const result = await this.echo.renderMessages(template, variables);
    // Extract text from the first message's content
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

  /** Get the raw template (from cache, Echostash, or defaults). */
  private async getTemplate(name: keyof PromptIds): Promise<string> {
    const cached = this.cache.get(name);
    if (cached) return cached;

    let template: string;
    if (this.client) {
      try {
        const prompt = await this.client.prompt(this.promptIds[name]).get();
        template = prompt.text();
      } catch {
        // Fallback to defaults on fetch error
        template = DEFAULTS[name];
      }
    } else {
      template = DEFAULTS[name];
    }

    this.cache.set(name, template);
    return template;
  }

  /** Clear the cache (forces re-fetch on next render). */
  clearCache(): void {
    this.cache.clear();
  }
}
