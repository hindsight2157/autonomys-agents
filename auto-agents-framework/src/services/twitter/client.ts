import { Scraper, SearchMode, Tweet, Profile } from 'agent-twitter-client';
import { createLogger } from '../../utils/logger.js';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { isValidTweet } from './convertFromTimeline.js';
import { convertTimelineTweetToTweet } from './convertFromTimeline.js';
import { TwitterApi } from './types.js';
const logger = createLogger('twitter-api');

const loadCookies = async (scraper: Scraper, cookiesPath: string): Promise<void> => {
  logger.info('Loading existing cookies');
  const cookies = readFileSync(cookiesPath, 'utf8');
  try {
    const parsedCookies = JSON.parse(cookies).map(
      (cookie: any) =>
        `${cookie.key}=${cookie.value}; Domain=${cookie.domain}; Path=${cookie.path}`,
    );
    await scraper.setCookies(parsedCookies);
    logger.info('Loaded existing cookies from file');
  } catch (error) {
    logger.error('Error loading cookies:', error);
    throw error;
  }
};

const login = async (
  scraper: Scraper,
  username: string,
  password: string,
  cookiesPath: string,
): Promise<void> => {
  logger.info('No existing cookies found, proceeding with login');
  await scraper.login(username, password);

  const newCookies = await scraper.getCookies();
  writeFileSync(cookiesPath, JSON.stringify(newCookies, null, 2));
  logger.info('New cookies saved to file');
};

const iterateResponse = async <T>(response: AsyncGenerator<T>): Promise<T[]> => {
  const iterated: T[] = [];
  for await (const item of response) {
    iterated.push(item);
  }
  return iterated;
};

const getUserReplyIds = async (
  scraper: Scraper,
  username: string,
  maxRepliesToCheck = 100,
): Promise<Set<string>> => {
  const replyIdSet = new Set<string>();

  // Query: all recent tweets from the user that are replies to someone else
  const userRepliesIterator = scraper.searchTweets(
    `from:${username} @`, // "from: user" + "@" ensures it's a reply/mention
    maxRepliesToCheck,
    SearchMode.Latest,
  );

  for await (const reply of userRepliesIterator) {
    if (reply.inReplyToStatusId) {
      replyIdSet.add(reply.inReplyToStatusId);
    }
  }

  return replyIdSet;
};

const getMyRecentReplies = async (
  scraper: Scraper,
  username: string,
  maxResults: number = 10,
): Promise<Tweet[]> => {
  const userRepliesIterator = scraper.searchTweets(
    `from:${username}`,
    maxResults,
    SearchMode.Latest,
  );
  const replies: Tweet[] = [];
  try {
    for await (const reply of userRepliesIterator) {
      if (replies.length >= maxResults) break;
      replies.push(reply);
    }
  } catch (error) {
    logger.error('Error fetching replies:', error);
  }
  return replies;
};

const getMyUnrepliedToMentions = async (
  scraper: Scraper,
  username: string,
  maxResults: number = 50,
  sinceId?: string,
): Promise<Tweet[]> => {
  const conversationCache = new Map<string, Tweet[]>();

  //TODO: This is not the way to get the thread, it is just a quick fix
  const getThread = async (scraper: Scraper, tweetId: string): Promise<Tweet[]> => {
    const initialTweet = await scraper.getTweet(tweetId);

    if (!initialTweet) {
      logger.warn(`Tweet ${tweetId} not found or deleted`);
      return [];
    }

    const conversationId = initialTweet.conversationId || initialTweet.id;

    // Check cache first
    const cachedConversation = conversationCache.get(conversationId!);
    if (cachedConversation) {
      return cachedConversation;
    }

    const conversationTweets = new Map<string, Tweet>();
    let rootTweet = initialTweet;

    // If the conversation root differs
    if (initialTweet.conversationId && initialTweet.conversationId !== initialTweet.id) {
      const conversationRoot = await scraper.getTweet(initialTweet.conversationId);
      if (conversationRoot) {
        rootTweet = conversationRoot;
        conversationTweets.set(rootTweet.id!, rootTweet);
        logger.info('Found conversation root tweet:', {
          id: rootTweet.id,
          conversationId: rootTweet.conversationId,
        });
      }
    } else {
      conversationTweets.set(rootTweet.id!, rootTweet);
    }

    try {
      //TODO: This does not return direct replies to the loggedin user, not sure why. Will need to investigate later
      const conversationIterator = scraper.searchTweets(
        `conversation_id:${conversationId}`,
        100,
        SearchMode.Latest,
      );
      for await (const tweet of conversationIterator) {
        conversationTweets.set(tweet.id!, tweet);
      }
    } catch (error) {
      logger.warn(`Error fetching conversation: ${error}`);
      return [rootTweet, initialTweet];
    }

    const thread = Array.from(conversationTweets.values());
    conversationCache.set(conversationId!, Array.from(conversationTweets.values()));
    return thread;
  };

  logger.info('Getting my mentions', { username, maxResults, sinceId });

  // get all mentions of the user (excluding user’s own tweets)
  const query = `@${username} -from:${username}`;
  const mentionIterator = scraper.searchTweets(query, maxResults, SearchMode.Latest);

  // build a set of "already replied to" tweet IDs in one query
  const repliedToIds = await getUserReplyIds(scraper, username, 100);

  // filter out any mention we've already replied to
  const newMentions: Tweet[] = [];
  for await (const tweet of mentionIterator) {
    // Stop if we've reached or passed the sinceId
    if (sinceId && tweet.id && tweet.id <= sinceId) {
      break;
    }

    // Skip if user has already replied
    if (repliedToIds.has(tweet.id!)) {
      logger.info(`Skipping tweet ${tweet.id} (already replied)`);
      continue;
    }

    newMentions.push(tweet);

    // Stop if we already have enough
    if (newMentions.length >= maxResults) {
      break;
    }
  }

  const withThreads = await Promise.all(
    newMentions.map(async mention => {
      const thread = await getThread(scraper, mention.id!);
      return {
        ...mention,
        thread,
      };
    }),
  );

  return withThreads;
};

const getFollowingRecentTweets = async (
  scraper: Scraper,
  username: string,
  maxResults: number = 50,
  randomNumberOfUsers: number = 10,
): Promise<Tweet[]> => {
  logger.info('Getting following recent tweets', {
    username,
    maxResults,
    randomNumberOfUsers,
  });
  const userId = await scraper.getUserIdByScreenName(username);
  const following = await iterateResponse(scraper.getFollowing(userId, 100));
  const randomFollowing = [...following]
    .sort(() => 0.5 - Math.random())
    .slice(0, randomNumberOfUsers);

  logger.info('Random Following', {
    randomFollowing: randomFollowing.map(user => user.username),
  });

  const query = `(${randomFollowing.map(user => `from:${user.username}`).join(' OR ')})`;
  const tweets = await iterateResponse(scraper.searchTweets(query, maxResults, SearchMode.Latest));
  return tweets;
};

export const createTwitterApi = async (
  username: string,
  password: string,
  cookiesPath: string = 'cookies.json',
): Promise<TwitterApi> => {
  const scraper = new Scraper();

  // Initialize authentication
  if (existsSync(cookiesPath)) {
    await loadCookies(scraper, cookiesPath);
  } else {
    await login(scraper, username, password, cookiesPath);
  }

  const isLoggedIn = await scraper.isLoggedIn();
  logger.info(`Login status: ${isLoggedIn}`);

  if (!isLoggedIn) {
    throw new Error('Failed to initialize Twitter Api - not logged in');
  }
  const userId = await scraper.getUserIdByScreenName(username);
  return {
    scraper,
    username: username,
    userId: userId,
    getMyUnrepliedToMentions: (maxResults: number, sinceId?: string) =>
      getMyUnrepliedToMentions(scraper, username, maxResults, sinceId),

    getFollowingRecentTweets: (maxResults: number = 100, randomNumberOfUsers: number = 10) =>
      getFollowingRecentTweets(scraper, username, maxResults, randomNumberOfUsers),

    isLoggedIn: () => scraper.isLoggedIn(),

    getProfile: async (username: string) => {
      const profile = await scraper.getProfile(username);
      if (!profile) {
        throw new Error(`Profile not found: ${username}`);
      }
      return profile;
    },

    getMyProfile: async () => await scraper.getProfile(username),

    getTweet: async (tweetId: string) => scraper.getTweet(tweetId),

    getRecentTweets: async (username: string, limit: number = 100) => {
      const userId = await scraper.getUserIdByScreenName(username);
      return await iterateResponse(scraper.getTweetsByUserId(userId, limit));
    },

    getMyRecentTweets: async (limit: number = 10) =>
      await iterateResponse(
        scraper.getTweetsByUserId(await scraper.getUserIdByScreenName(username), limit),
      ),

    getMyRepliedToIds: async () => Array.from(await getUserReplyIds(scraper, username, 100)),

    getFollowing: async (userId: string, limit: number = 100) =>
      await iterateResponse(scraper.getFollowing(userId, limit)),

    getMyTimeline: async (count: number, excludeIds: string[]) => {
      const tweets = await scraper.fetchHomeTimeline(count, excludeIds);
      return tweets.filter(isValidTweet).map(tweet => convertTimelineTweetToTweet(tweet));
    },

    getFollowingTimeline: async (count: number, excludeIds: string[]) => {
      const tweets = await scraper.fetchFollowingTimeline(count, excludeIds);
      return tweets.filter(isValidTweet).map(tweet => convertTimelineTweetToTweet(tweet));
    },

    getMyRecentReplies: (limit: number = 10) => getMyRecentReplies(scraper, username, limit),

    //TODO: After sending the tweet, we need to get the latest tweet, ensure it is the same as we sent and return it
    //This has not been working as expected, so we need to investigate this later
    sendTweet: async (tweet: string, inReplyTo?: string) => {
      tweet.length > 280
        ? await scraper.sendLongTweet(tweet, inReplyTo)
        : await scraper.sendTweet(tweet, inReplyTo);
      logger.info('Tweet sent', { tweet, inReplyTo });
      getMyRecentReplies;
    },
  };
};
