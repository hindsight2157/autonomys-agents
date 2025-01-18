import inquirer from 'inquirer';
import { createLogger } from '../utils/logger.js';
import { listAvailableCharacters } from './utils/characterLoader.js';

const logger = createLogger('onboarding');

interface UserAnswers {
  character: string;
}

export const onboarding = async (): Promise<UserAnswers> => {
  const characters = await listAvailableCharacters();

  const answers: UserAnswers = await inquirer.prompt([
    {
      type: 'list',
      name: 'character',
      message: 'Select a character to run the workflow:',
      choices: characters.map(char => ({
        name: `${char.id} - ${char.description.split('.')[0]}`,
        value: char.id,
      })),
    },
  ]);
  logger.info(`Character: ${answers.character}`);

  return answers;
};
