# refine the incorrect api calls
import json
from time import sleep


def main(use_chat: bool, filename):
    from utils.llm.openai_prompt import load_api_key_from_envion
    load_api_key_from_envion()

    for json_str in open(filename):
        new_json_str = refine_line(use_chat, json_str)
        print(new_json_str)
        sleep(20)


def refine_line(use_chat, json_str):
    from utils.llm.openai_prompt import chat_completion, completion
    obj = json.loads(json_str)
    api_success = obj['api_success']
    if api_success:
        return json_str

    try:
        res = chat_completion(obj['prompt']) if use_chat else completion(obj['prompt'])
        success = True
    except:
        res = ''
        success = False

    obj['api_success'] = success
    obj['pred'] = res
    obj['correct'] = obj['tgt'] == res
    return json.dumps(obj)


if __name__ == '__main__':
    import sys
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC'))
    use_chat = sys.argv[1] == 'chat'
    in_file = sys.argv[2]
    main(use_chat, in_file)

