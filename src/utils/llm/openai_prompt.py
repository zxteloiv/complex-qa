import openai
import logging

logger = logging.getLogger(__name__)


def load_api_key_from_envion():
    import os
    key = os.getenv('OPENAI_API_KEY')
    if key is None:
        raise ValueError('No OPENAI_API_KEY env variable set.')
    else:
        openai.api_key = key


def completion(prompt: str, model='text-davinci-003') -> str:
    logger.debug(f'Using {model} for Completion: '
                 f'{prompt[:10] + "..." + prompt[-10:] if len(prompt) > 20 else prompt}')

    res = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=.0,
    )
    return res['choices'][0]['text'].strip()


def chat_completion(prompt: str, model='gpt-3.5-turbo') -> str:
    logger.debug(f'Using {model} for ChatCompletion: '
                 f'{prompt[:10] + "..." + prompt[-10:] if len(prompt) > 20 else prompt}')

    res = openai.ChatCompletion.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        temperature=.0,
    )
    return res['choices'][0]['message']['content']
