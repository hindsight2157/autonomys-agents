import { WorkflowConfig } from '../types.js';
import { createLogger } from '../../../../utils/logger.js';
import { State } from '../workflow.js';
import { summaryParser, summaryPrompt } from '../prompts.js';

const logger = createLogger('summary-node');

export const createSummaryNode = (config: WorkflowConfig) => async (state: typeof State.State) => {
  logger.info('Summary Node - Summarizing trends');
  const myRecentReplies = Array.from(state.myRecentReplies.values()).map(reply => reply.text);

  const summary = await summaryPrompt
    .pipe(config.llms.analyze)
    .pipe(summaryParser)
    .invoke({
      tweets: JSON.stringify(myRecentReplies),
    });
  logger.info('Summary:', summary);
  return {
    summary,
  };
};
