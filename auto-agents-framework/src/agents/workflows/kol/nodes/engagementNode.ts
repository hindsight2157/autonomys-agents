import { AIMessage } from '@langchain/core/messages';
import { WorkflowConfig } from '../types.js';
import { createLogger } from '../../../../utils/logger.js';
import { State } from '../workflow.js';
import { engagementParser, engagementPrompt } from '../prompts.js';
import { Tweet } from '../../../../services/twitter/types.js';

const logger = createLogger('engagement-node');

const getEngagementDecision = async (tweet: Tweet, config: WorkflowConfig) => {
  const thread =
    tweet.thread && tweet.thread.length > 0
      ? tweet.thread.map(t => ({ text: t.text, username: t.username }))
      : 'No thread';

  const formattedPrompt = await engagementPrompt.format({
    tweet: JSON.stringify({ text: tweet.text, username: tweet.username }),
    thread: thread,
  });

  return await config.llms.decision.pipe(engagementParser).invoke(formattedPrompt);
};

export const createEngagementNode = (config: WorkflowConfig) => {
  return async (state: typeof State.State) => {
    logger.info('Engagement Node - Starting evaluation');
    try {
      const { mentionsTweets, timelineTweets } = state;
      const tweets = [...mentionsTweets, ...timelineTweets];
      const engagementDecisions = await Promise.all(
        tweets.map(async tweet => {
          const decision = await getEngagementDecision(tweet, config);
          logger.info('Engagement Decision', {
            tweet: tweet.text,
            thread: tweet.thread && tweet.thread.length > 0 ? tweet.thread[0].text : 'No thread',
            decision,
          });
          return {
            tweet: {
              id: tweet.id!,
              text: tweet.text!,
              username: tweet.username!,
              timeParsed: tweet.timeParsed!,
            },
            decision,
          };
        }),
      );
      return {
        messages: [
          new AIMessage({
            content: JSON.stringify({
              engagementDecisions,
            }),
          }),
        ],
        engagementDecisions: engagementDecisions,
      };
    } catch (error) {
      logger.error('Error in engagement node:', error);
      return { messages: [] };
    }
  };
};
